import math
from dataclasses import replace

import pytest
import torch

import reciprocator.training as training
from reciprocator.training import (
    _configure_tensor_dynamic_growth,
    _learning_rate_scale_for_step,
    _mode_prune_copy_state_dict,
    _pad_copy_state_dict,
    LispGRPOConfig,
    RLTrainingResult,
    TrainingConfig,
    build_lisp_tokenizer,
    build_corpus_dataset,
    build_text_dataset,
    sample_causal_lm_batch,
    train_lisp_grpo,
    train_model,
    TrainingResumeState,
)


def test_build_text_dataset_round_trips_characters() -> None:
    dataset = build_text_dataset("abca cab", val_fraction=0.25, min_split_tokens=4)

    assert dataset.vocab_size == len(set("abca cab"))
    decoded = dataset.tokenizer.decode(dataset.train_tokens.tolist() + dataset.val_tokens.tolist())
    assert decoded == "abca cab"


def test_training_module_exports_lisp_grpo_helpers() -> None:
    tokenizer = build_lisp_tokenizer()

    assert tokenizer.vocab_size > 10
    assert LispGRPOConfig().steps > 0
    assert LispGRPOConfig().readout_type == "phase_aware"
    assert RLTrainingResult.__name__ == "RLTrainingResult"
    assert callable(train_lisp_grpo)


def test_training_config_defaults_match_stronger_experiment_baseline() -> None:
    config = TrainingConfig()

    assert config.corpus_name == "greek_classics"
    assert config.max_chars == 100000
    assert config.batch_size == 8
    assert config.seq_len == 128
    assert config.steps == 2000
    assert config.eval_every == 100
    assert config.eval_batches == 4
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.hidden_size == 256
    assert config.state_shape == (4, 4, 4)
    assert config.readout_type == "phase_aware"
    assert config.token_magnitude_type == "inverse_frequency_learned"
    assert config.phase_type == "rope"
    assert config.token_phase == "semantic"
    assert config.coupling_type == "sequential"
    assert config.enable_self_relation is True
    assert config.dynamic_spectral_gains is False
    assert config.anisotropic_spectral_gains is False
    assert config.normalization_type == "frobenius"
    assert config.post_growth_cooldown_checks == 0
    assert config.post_growth_cooldown_threshold_scale == pytest.approx(1.5)
    assert config.num_layers == 1


def test_build_corpus_dataset_uses_bundled_corpus() -> None:
    dataset = build_corpus_dataset("greek_classics", max_chars=256, val_fraction=0.2)

    assert dataset.source_name == "greek_classics"
    assert dataset.train_tokens.numel() > dataset.val_tokens.numel()
    assert dataset.vocab_size > 10


def test_sample_causal_lm_batch_returns_shifted_targets() -> None:
    torch.manual_seed(0)
    tokens = torch.arange(32, dtype=torch.long)

    inputs, targets = sample_causal_lm_batch(tokens, batch_size=3, seq_len=5)

    assert inputs.shape == (3, 5)
    assert targets.shape == (3, 5)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert torch.all(targets[:, -1] == inputs[:, -1] + 1)


def test_train_model_smoke_runs() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=3,
        eval_every=2,
        eval_batches=2,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        device="cpu",
        seed=0,
    )

    result = train_model(config, dataset=dataset)

    assert len(result.train_losses) == 3
    assert len(result.train_metrics) == 3
    assert len(result.val_losses) == 3
    assert len(result.val_metrics) == 3
    assert result.generation_samples == []
    assert result.runtime_benchmarks == []
    assert torch.isfinite(torch.tensor(result.train_losses)).all()
    assert torch.isfinite(torch.tensor([loss for _, loss in result.val_losses])).all()
    assert all(0.0 <= metrics.accuracy <= 1.0 for _, metrics in result.train_metrics)
    assert all(0.0 <= metrics.accuracy <= 1.0 for _, metrics in result.val_metrics)
    assert [row["step"] for row in result.residual_diagnostics] == [1, 2, 3]
    assert all(row["state_shape"] == [2, 2] for row in result.residual_diagnostics)
    assert all(row["growth_event_history"] == [] for row in result.residual_diagnostics)
    assert all(row["cross_memory_residual"] == pytest.approx(0.0) for row in result.residual_diagnostics)
    assert all(row["token_count"] == config.seq_len for row in result.residual_diagnostics)
    assert all("mean_phase_variance" in row for row in result.residual_diagnostics)

    last_train_metrics = result.train_metrics[-1][1]
    last_val_metrics = result.val_metrics[-1][1]
    assert last_train_metrics.perplexity == pytest.approx(math.exp(last_train_metrics.loss))
    assert last_train_metrics.bpc == pytest.approx(last_train_metrics.loss / math.log(2.0))
    assert last_val_metrics.perplexity == pytest.approx(math.exp(last_val_metrics.loss))
    assert last_val_metrics.bpc == pytest.approx(last_val_metrics.loss / math.log(2.0))


def test_learning_rate_scale_supports_warmup_and_cosine_decay() -> None:
    config = TrainingConfig(
        steps=6,
        lr_warmup_steps=2,
        lr_decay_style="cosine",
        min_lr_scale=0.2,
    )

    assert _learning_rate_scale_for_step(config, 1) == pytest.approx(0.5)
    assert _learning_rate_scale_for_step(config, 2) == pytest.approx(1.0)
    assert _learning_rate_scale_for_step(config, 3) == pytest.approx(0.882842712474619)
    assert _learning_rate_scale_for_step(config, 6) == pytest.approx(0.2)


def test_train_model_applies_weight_decay_and_gradient_clipping(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=1,
        eval_every=1,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        grad_clip_norm=1.25,
        weight_decay=0.03,
        device="cpu",
        seed=0,
    )

    optimizer_weight_decays = []
    clipped_norms = []
    real_adam = training.torch.optim.Adam
    real_clip_grad_norm = training.torch.nn.utils.clip_grad_norm_

    def capture_adam(params, *args, **kwargs):
        optimizer_weight_decays.append(kwargs.get("weight_decay"))
        return real_adam(params, *args, **kwargs)

    def capture_clip_grad_norm(parameters, max_norm, *args, **kwargs):
        clipped_norms.append(max_norm)
        return real_clip_grad_norm(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(training.torch.optim, "Adam", capture_adam)
    monkeypatch.setattr(training.torch.nn.utils, "clip_grad_norm_", capture_clip_grad_norm)

    train_model(config, dataset=dataset)

    assert optimizer_weight_decays == [pytest.approx(0.03)]
    assert clipped_norms == [pytest.approx(1.25)]


def test_train_model_records_generation_samples_and_runtime_benchmarks() -> None:
    dataset = build_text_dataset("abcde " * 60, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=2,
        eval_every=1,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        generation_eval_samples=2,
        generation_prompt_len=8,
        generation_new_tokens=10,
        benchmark_prompt_lengths=(4, 8),
        benchmark_new_tokens=6,
        device="cpu",
        seed=0,
    )

    result = train_model(config, dataset=dataset)

    assert len(result.generation_samples) == 2
    assert all(len(sample.continuation) == 10 for sample in result.generation_samples)
    assert all(0.0 <= sample.distinct_1 <= 1.0 for sample in result.generation_samples)
    assert all(0.0 <= sample.distinct_2 <= 1.0 for sample in result.generation_samples)
    assert [benchmark.prompt_length for benchmark in result.runtime_benchmarks] == [4, 8]
    assert all(benchmark.decode_tokens_per_second > 0.0 for benchmark in result.runtime_benchmarks)
    assert all(benchmark.memory_metric for benchmark in result.runtime_benchmarks)


def test_state_signal_features_are_phase_invariant_and_compact() -> None:
    state_signal = torch.complex(
        torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4),
        torch.arange(25, 49, dtype=torch.float32).reshape(2, 3, 4),
    )
    global_phase = torch.polar(torch.ones(1, 1, 1), torch.tensor(0.41))

    features = training._state_signal_features(state_signal)
    rotated_features = training._state_signal_features(state_signal * global_phase)

    assert features.shape == (3, 2, 3, 4)
    assert torch.allclose(rotated_features, features, atol=1e-5)


def test_train_model_supports_per_mode_normalization() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=2,
        eval_every=1,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        normalization_type="per_mode",
        device="cpu",
        seed=0,
    )

    result = train_model(config, dataset=dataset)

    assert result.model.normalization_type == "per_mode"
    assert torch.isfinite(torch.tensor(result.train_losses)).all()


def test_train_model_supports_cross_layer_state() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=2,
        eval_every=1,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=2,
        enable_cross_layer_state=True,
        device="cpu",
        seed=0,
    )

    result = train_model(config, dataset=dataset)

    assert result.model.enable_cross_layer_state is True
    assert len(result.model.cross_layer_proj) == 1
    assert len(result.model.cross_layer_beta) == 1
    assert torch.isfinite(torch.tensor(result.train_losses)).all()


def test_train_model_supports_fft_coupling() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=2,
        eval_every=1,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        coupling_type="fft",
        dynamic_spectral_gains=True,
        anisotropic_spectral_gains=True,
        device="cpu",
        seed=0,
    )

    result = train_model(config, dataset=dataset)

    assert result.model.coupling_type == "fft"
    assert result.model.dynamic_spectral_gains is True
    assert result.model.anisotropic_spectral_gains is True
    assert all(block.mixer.coupling_type == "fft" for block in result.model.blocks)
    assert all(block.mixer.coupling.dynamic_spectral_gains for block in result.model.blocks)
    assert all(block.mixer.coupling.anisotropic_spectral_gains for block in result.model.blocks)
    assert torch.isfinite(torch.tensor(result.train_losses)).all()


def test_train_model_invokes_step_callback_each_step() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=4,
        eval_every=2,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        device="cpu",
        seed=0,
    )
    seen_steps = []

    def callback(
        step,
        model,
        optimizer,
        callback_dataset,
        callback_config,
        train_losses,
        val_losses,
        train_metrics,
        val_metrics,
        device,
    ) -> None:
        seen_steps.append(step)
        assert callback_dataset is dataset
        assert callback_config is config
        assert len(train_losses) == step
        assert len(train_metrics) == step

    train_model(config, dataset=dataset, step_callback=callback)

    assert seen_steps == [1, 2, 3, 4]


def test_train_model_resumes_from_checkpoint_state(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 50, val_fraction=0.2, min_split_tokens=16)
    initial_config = TrainingConfig(
        steps=2,
        eval_every=1,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        stateful_training=False,
        device="cpu",
        seed=0,
    )
    initial_result = train_model(initial_config, dataset=dataset)
    optimizer_state = initial_result.optimizer.state_dict()
    resume_state = TrainingResumeState(
        step=2,
        model_state_dict=initial_result.model.state_dict(),
        optimizer_state_dict=optimizer_state,
        train_losses=initial_result.train_losses,
        val_losses=initial_result.val_losses,
        train_metrics=initial_result.train_metrics,
        val_metrics=initial_result.val_metrics,
    )
    resumed_config = replace(initial_config, steps=4)
    seen_steps = []
    loaded_optimizer_states = []
    real_load_state_dict = training.torch.optim.Adam.load_state_dict

    def capture_load_state_dict(self, state_dict):
        loaded_optimizer_states.append(state_dict)
        return real_load_state_dict(self, state_dict)

    monkeypatch.setattr(training.torch.optim.Adam, "load_state_dict", capture_load_state_dict)

    def callback(
        step,
        model,
        optimizer,
        callback_dataset,
        callback_config,
        train_losses,
        val_losses,
        train_metrics,
        val_metrics,
        device,
    ) -> None:
        seen_steps.append(step)
        assert callback_config is resumed_config
        assert len(train_losses) == step
        assert len(train_metrics) == step

    resumed_result = train_model(
        resumed_config,
        dataset=dataset,
        step_callback=callback,
        resume_state=resume_state,
    )

    assert seen_steps == [3, 4]
    assert len(loaded_optimizer_states) == 1
    assert loaded_optimizer_states[0] is optimizer_state
    assert resumed_result.train_losses[:2] == initial_result.train_losses
    assert resumed_result.val_losses[:2] == initial_result.val_losses
    assert resumed_result.train_metrics[:2] == initial_result.train_metrics
    assert resumed_result.val_metrics[:2] == initial_result.val_metrics
    assert len(resumed_result.train_losses) == 4
    assert resumed_result.config.state_shape == (2, 2)


def test_configure_tensor_dynamic_growth_enables_cuda_expandable_segments(monkeypatch) -> None:
    seen_settings = []

    monkeypatch.setattr(
        torch.cuda.memory,
        "_set_allocator_settings",
        lambda settings: seen_settings.append(settings),
    )

    _configure_tensor_dynamic_growth(torch.device("cuda"), enabled=True)

    assert seen_settings == ["expandable_segments:True"]


def test_configure_tensor_dynamic_growth_skips_non_cuda_or_disabled(monkeypatch) -> None:
    seen_settings = []

    monkeypatch.setattr(
        torch.cuda.memory,
        "_set_allocator_settings",
        lambda settings: seen_settings.append(settings),
    )

    _configure_tensor_dynamic_growth(torch.device("cpu"), enabled=True)
    _configure_tensor_dynamic_growth(torch.device("cuda"), enabled=False)

    assert seen_settings == []


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("growth_residual_threshold", -0.1),
        ("residual_saturate_threshold", -0.1),
        ("growth_residual_ema_decay", -0.1),
        ("growth_residual_ema_decay", 1.0),
        ("post_growth_cooldown_checks", -1),
        ("post_growth_cooldown_threshold_scale", 0.99),
        ("chunk_size", 0),
        ("min_checks_before_first_growth", -1),
        ("rank_growth_loss_ceiling", -0.1),
        ("prune_threshold", -0.1),
        ("prune_sustain_steps", -1),
        ("prune_min_steps", -1),
    ],
)
def test_train_model_rejects_invalid_residual_growth_config(field_name: str, field_value: float) -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(**{field_name: field_value})

    with pytest.raises(ValueError, match=field_name):
        train_model(config, dataset=dataset)


def test_train_model_runs_config_validation_before_stateful_stream_validation() -> None:
    dataset = build_text_dataset("abcde " * 24, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        batch_size=12,
        seq_len=10,
        attention_window=0,
        stateful_training=True,
    )

    with pytest.raises(ValueError, match="attention_window"):
        train_model(config, dataset=dataset)


def test_train_model_allows_dynamic_growth_for_spectral_coupling() -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=1,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        device="cpu",
        coupling_type="dwt",
        dynamic_mode_growth=True,
    )

    result = train_model(config, dataset=dataset)

    assert result.config.coupling_type == "dwt"


def test_train_model_respects_min_checks_before_first_growth_before_attempting_dynamic_growth(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=6,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        device="cpu",
        dynamic_mode_growth=True,
        growth_check_interval=2,
        min_checks_before_first_growth=2,
        seed=0,
    )

    growth_attempt_steps = []

    def fake_try_dynamic_growth(
        model,
        callback_config,
        callback_dataset,
        device,
        recent_losses,
        smoothed_mode_residual_norms,
        reference_token_ids=None,
        effective_growth_residual_threshold=None,
    ):
        growth_attempt_steps.append(len(recent_losses) + 1)
        return None, None

    monkeypatch.setattr(training, "_try_dynamic_growth", fake_try_dynamic_growth)

    train_model(config, dataset=dataset)

    assert growth_attempt_steps == [4, 6]


def test_train_model_logs_growth_event_history_at_validation(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=2,
        eval_every=1,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        device="cpu",
        dynamic_mode_growth=True,
        growth_check_interval=1,
        seed=0,
    )
    grown_config = replace(config, state_shape=(3, 2))
    grown_model = training._build_model_from_config(grown_config, dataset, torch.device("cpu"))
    growth_results = iter([(grown_model, grown_config), (None, None)])

    monkeypatch.setattr(training, "_compute_mode_residual_norms", lambda model, token_ids: torch.tensor([1.0, 1.0]))
    monkeypatch.setattr(training, "_try_dynamic_growth", lambda *args, **kwargs: next(growth_results))

    result = train_model(config, dataset=dataset)

    assert result.residual_diagnostics[0]["growth_event_history"] == [(1, [2, 2], [3, 2])]
    assert result.residual_diagnostics[-1]["growth_event_history"] == [(1, [2, 2], [3, 2])]


def test_train_model_applies_post_growth_cooldown_threshold_for_next_checks(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=4,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        device="cpu",
        dynamic_mode_growth=True,
        growth_check_interval=1,
        growth_residual_threshold=0.4,
        post_growth_cooldown_checks=2,
        post_growth_cooldown_threshold_scale=1.5,
        seed=0,
    )
    grown_config = replace(config, state_shape=(3, 2))
    grown_model = training._build_model_from_config(grown_config, dataset, torch.device("cpu"))
    effective_thresholds = []

    def fake_try_dynamic_growth(
        model,
        callback_config,
        callback_dataset,
        device,
        recent_losses,
        smoothed_mode_residual_norms,
        reference_token_ids=None,
        effective_growth_residual_threshold=None,
    ):
        effective_thresholds.append(effective_growth_residual_threshold)
        if len(effective_thresholds) == 1:
            return grown_model, grown_config
        return None, None

    monkeypatch.setattr(training, "_compute_mode_residual_norms", lambda model, token_ids: torch.tensor([1.0, 1.0]))
    monkeypatch.setattr(training, "_try_dynamic_growth", fake_try_dynamic_growth)

    train_model(config, dataset=dataset)

    assert effective_thresholds == pytest.approx([0.4, 0.6, 0.6, 0.4])


def test_train_model_records_residual_diagnostics_on_growth_check_cadence(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=4,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        device="cpu",
        growth_check_interval=2,
        record_residual_diagnostics=True,
        seed=0,
    )

    residual_sequence = iter(
        [
            torch.tensor([1.0, 0.5]),
            torch.tensor([0.0, 1.0]),
        ]
    )
    redundancy_sequence = iter(
        [
            torch.tensor([0.25, 0.75]),
            torch.tensor([0.5, 0.0]),
        ]
    )
    slice_variance_sequence = iter(
        [
            [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
            [torch.tensor([0.5, 0.6]), torch.tensor([0.7, 0.8])],
        ]
    )
    residual_call_count = 0
    redundancy_call_count = 0
    slice_variance_call_count = 0

    def fake_compute_mode_residual_norms(model, token_ids):
        nonlocal residual_call_count
        residual_call_count += 1
        return next(residual_sequence)

    def fake_compute_mode_pruning_residual_norms(model, token_ids):
        nonlocal redundancy_call_count
        redundancy_call_count += 1
        return next(redundancy_sequence)

    def fake_compute_mode_slice_activation_variances(model, token_ids):
        nonlocal slice_variance_call_count
        slice_variance_call_count += 1
        return next(slice_variance_sequence)

    monkeypatch.setattr(training, "_compute_mode_residual_norms", fake_compute_mode_residual_norms)
    monkeypatch.setattr(training, "_compute_mode_pruning_residual_norms", fake_compute_mode_pruning_residual_norms)
    monkeypatch.setattr(training, "_compute_mode_slice_activation_variances", fake_compute_mode_slice_activation_variances)

    result = train_model(config, dataset=dataset)

    assert residual_call_count == 2
    assert redundancy_call_count == 2
    assert slice_variance_call_count == 2
    assert [row["step"] for row in result.residual_diagnostics] == [1, 2, 4]
    rows_by_step = {row["step"]: row for row in result.residual_diagnostics}
    assert rows_by_step[2]["mode_residual_ema"] == pytest.approx([1.0, 0.5])
    assert rows_by_step[4]["mode_residual_ema"] == pytest.approx([0.95, 0.525])
    assert rows_by_step[2]["mode_redundancy_ema"] == pytest.approx([0.25, 0.75])
    assert rows_by_step[4]["mode_redundancy_ema"] == pytest.approx([0.2625, 0.7125])
    assert rows_by_step[2]["mode_slice_activation_variance_ema"][0] == pytest.approx([0.1, 0.2])
    assert rows_by_step[2]["mode_slice_activation_variance_ema"][1] == pytest.approx([0.3, 0.4])
    assert rows_by_step[4]["mode_slice_activation_variance_ema"][0] == pytest.approx([0.12, 0.22])
    assert rows_by_step[4]["mode_slice_activation_variance_ema"][1] == pytest.approx([0.32, 0.42])
    assert rows_by_step[2]["prune_candidate_streaks"] == [1, 0]
    assert rows_by_step[4]["prune_candidate_streaks"] == [2, 0]
    assert rows_by_step[1]["cross_memory_residual"] == pytest.approx(0.0)
    assert rows_by_step[4]["val_bpc"] == pytest.approx(result.val_metrics[-1][1].bpc)


def test_train_model_records_layer_mode_residual_diagnostics_for_multilayer_models(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=4,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=2,
        device="cpu",
        growth_check_interval=2,
        record_residual_diagnostics=True,
        seed=0,
    )

    aggregate_sequence = iter(
        [
            torch.tensor([0.6, 0.4]),
            torch.tensor([0.4, 0.8]),
        ]
    )
    layer_sequence = iter(
        [
            torch.tensor([[1.0, 0.5], [0.2, 0.3]]),
            torch.tensor([[0.0, 1.0], [0.8, 0.6]]),
        ]
    )

    monkeypatch.setattr(training, "_compute_mode_residual_norms", lambda model, token_ids: next(aggregate_sequence))
    monkeypatch.setattr(training, "_compute_layer_mode_residual_norms", lambda model, token_ids: next(layer_sequence))

    result = train_model(config, dataset=dataset)

    rows_by_step = {row["step"]: row for row in result.residual_diagnostics}
    assert rows_by_step[2]["layer_mode_residual_norms"][0] == pytest.approx([1.0, 0.5])
    assert rows_by_step[2]["layer_mode_residual_norms"][1] == pytest.approx([0.2, 0.3])
    assert rows_by_step[2]["layer_mode_residual_ema"][0] == pytest.approx([1.0, 0.5])
    assert rows_by_step[2]["layer_mode_residual_ema"][1] == pytest.approx([0.2, 0.3])
    assert rows_by_step[4]["layer_mode_residual_norms"][0] == pytest.approx([0.0, 1.0])
    assert rows_by_step[4]["layer_mode_residual_norms"][1] == pytest.approx([0.8, 0.6])
    assert rows_by_step[4]["layer_mode_residual_ema"][0] == pytest.approx([0.95, 0.525])
    assert rows_by_step[4]["layer_mode_residual_ema"][1] == pytest.approx([0.23, 0.315])


def test_train_model_records_residual_diagnostics_for_hybrid_attention_models() -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=2,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=2,
        device="cpu",
        growth_check_interval=1,
        record_residual_diagnostics=True,
        attention_every_k=1,
        attention_num_heads=3,
        attention_window=8,
        attention_position="after",
        seed=0,
    )

    result = train_model(config, dataset=dataset)

    assert any(not hasattr(block, "mixer") for block in result.model.blocks)
    assert [row["step"] for row in result.residual_diagnostics] == [1, 2]
    assert len(result.residual_diagnostics[0]["layer_mode_residual_norms"]) == 2
    assert math.isfinite(result.residual_diagnostics[0]["cross_memory_residual"])
    assert result.residual_diagnostics[0]["cross_memory_residual"] >= 0.0
    assert [len(mode_variances) for mode_variances in result.residual_diagnostics[0]["mode_slice_activation_variance_ema"]] == [2, 2]
    assert all(
        math.isfinite(value) and value >= 0.0
        for mode_variances in result.residual_diagnostics[0]["mode_slice_activation_variance_ema"]
        for value in mode_variances
    )


def test_train_model_records_chunk_drift_history_when_requested() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=2,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        chunk_size=4,
        track_chunk_drift=True,
        device="cpu",
        seed=0,
    )

    result = train_model(config, dataset=dataset)

    assert len(result.chunk_drift_history) == 2
    assert [row["step"] for row in result.chunk_drift_history] == [1, 2]
    assert all(row["K"] == 4 for row in result.chunk_drift_history)
    assert all(row["mean_drift"] >= 0.0 for row in result.chunk_drift_history)
    assert all(row["max_drift"] >= row["mean_drift"] for row in result.chunk_drift_history)


def test_train_model_warns_when_recent_chunk_drift_is_high(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=4,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        chunk_size=4,
        track_chunk_drift=True,
        growth_check_interval=2,
        device="cpu",
        seed=0,
    )

    monkeypatch.setattr(training, "_compute_mode_residual_norms", lambda model, token_ids: torch.tensor([0.1, 0.1]))

    def fake_compute_batch_metrics(
        model,
        token_ids,
        targets,
        *,
        states=None,
        chunk_size=None,
        track_drift=False,
    ):
        return (
            torch.tensor(0.0, requires_grad=True),
            training._build_metrics(0.0, 0.0),
            tuple(None for _ in model.blocks),
            {
                "mean_drift": training.CHUNK_DRIFT_WARN_THRESHOLD + 0.01,
                "max_drift": training.CHUNK_DRIFT_WARN_THRESHOLD + 0.02,
                "K": chunk_size,
            }
            if track_drift
            else None,
        )

    monkeypatch.setattr(training, "_compute_batch_metrics", fake_compute_batch_metrics)

    with pytest.warns(UserWarning, match="Mean chunk drift"):
        train_model(config, dataset=dataset)


def test_train_model_forces_exact_forward_after_growth_rebuild(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 20, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        steps=4,
        eval_every=10,
        eval_batches=1,
        batch_size=4,
        seq_len=8,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        chunk_size=4,
        growth_check_interval=2,
        dynamic_mode_growth=True,
        min_checks_before_first_growth=1,
        stateful_training=False,
        device="cpu",
        seed=0,
    )

    rebuilt_model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    growth_results = iter([(rebuilt_model, config), (None, None)])
    seen_chunk_sizes = []

    monkeypatch.setattr(training, "_compute_mode_residual_norms", lambda model, token_ids: torch.tensor([1.0, 1.0]))
    monkeypatch.setattr(training, "_try_dynamic_growth", lambda *args, **kwargs: next(growth_results))

    def fake_compute_batch_metrics(
        model,
        token_ids,
        targets,
        *,
        states=None,
        chunk_size=None,
        track_drift=False,
    ):
        seen_chunk_sizes.append(chunk_size)
        return (
            torch.tensor(0.0, requires_grad=True),
            training._build_metrics(0.0, 0.0),
            tuple(None for _ in model.blocks),
            None,
        )

    monkeypatch.setattr(training, "_compute_batch_metrics", fake_compute_batch_metrics)

    train_model(config, dataset=dataset)

    assert seen_chunk_sizes == [4, None, None, 4]


def test_pad_copy_state_dict_supports_rank_growth() -> None:
    old_tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    merged = _pad_copy_state_dict(
        {"tensor": old_tensor},
        {"tensor": torch.zeros(2, 3, 4, 2, dtype=torch.float32)},
    )

    assert torch.equal(merged["tensor"][..., 0], old_tensor)
    assert torch.count_nonzero(merged["tensor"][..., 1]) == 0


def test_pad_copy_state_dict_supports_rank_growth_residual_reference() -> None:
    old_decay = torch.ones(2, 3, 4, dtype=torch.float32)
    residual_state_signal = torch.complex(
        torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4),
        torch.zeros(2, 3, 4, dtype=torch.float32),
    )
    merged = _pad_copy_state_dict(
        {"blocks.0.mixer.decay_logit": old_decay},
        {"blocks.0.mixer.decay_logit": torch.zeros(2, 3, 4, 2, dtype=torch.float32)},
        old_state_shape=(2, 3, 4),
        new_state_shape=(2, 3, 4, 2),
        growth_kind="rank",
        grown_axis=3,
        rank_init="residual",
        reference_contexts=[
            {
                "hidden_residual_features": torch.ones(4),
                "rank_residual_state_signal": residual_state_signal,
                "rank_residual_state_features": training._state_signal_features(residual_state_signal),
            }
        ],
    )

    expected = residual_state_signal.abs()
    expected = expected / expected.norm() * old_decay.reshape(1, -1).norm(dim=1).mean()

    assert torch.equal(merged["blocks.0.mixer.decay_logit"][..., 0], old_decay)
    assert torch.allclose(merged["blocks.0.mixer.decay_logit"][..., 1], expected)


def test_pad_copy_state_dict_supports_mode_growth_residual_reference() -> None:
    old_decay = torch.ones(2, 3, 4, dtype=torch.float32)
    state_signal = torch.complex(
        torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4),
        torch.zeros(2, 3, 4, dtype=torch.float32),
    )
    merged = _pad_copy_state_dict(
        {"blocks.0.mixer.decay_logit": old_decay},
        {"blocks.0.mixer.decay_logit": torch.zeros(3, 3, 4, dtype=torch.float32)},
        old_state_shape=(2, 3, 4),
        new_state_shape=(3, 3, 4),
        growth_kind="mode",
        grown_axis=0,
        mode_init="residual",
        reference_contexts=[
            {
                "hidden_features": torch.ones(4),
                "state_signal": state_signal,
                "state_signal_features": training._state_signal_features(state_signal),
            }
        ],
    )

    expected = state_signal.abs().mean(dim=0)
    expected = expected / expected.norm() * old_decay.movedim(0, 0).reshape(2, -1).norm(dim=1).mean()

    assert torch.equal(merged["blocks.0.mixer.decay_logit"][:2], old_decay)
    assert torch.allclose(merged["blocks.0.mixer.decay_logit"][2], expected)


def test_pad_copy_state_dict_supports_mode_growth_for_return_map_weights() -> None:
    old_weight = torch.arange(72 * 8, dtype=torch.float32).reshape(8, 72)
    state_signal = torch.complex(
        torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4),
        torch.zeros(2, 3, 4, dtype=torch.float32),
    )
    merged = _pad_copy_state_dict(
        {"blocks.0.mixer.return_map.proj.weight": old_weight},
        {"blocks.0.mixer.return_map.proj.weight": torch.zeros(8, 108, dtype=torch.float32)},
        old_state_shape=(2, 3, 4),
        new_state_shape=(3, 3, 4),
        growth_kind="mode",
        grown_axis=0,
        mode_init="residual",
        reference_contexts=[
            {
                "hidden_features": torch.ones(8),
                "state_signal": state_signal,
                "state_signal_features": training._state_signal_features(state_signal),
            }
        ],
    )

    merged_view = merged["blocks.0.mixer.return_map.proj.weight"].reshape(8, 3, 3, 3, 4)
    old_view = old_weight.reshape(8, 3, 2, 3, 4)

    assert torch.equal(merged_view[:, :, :2], old_view)
    assert torch.count_nonzero(merged_view[:, :, 2]) > 0


def test_pad_copy_state_dict_supports_cross_layer_proj_weights() -> None:
    old_weight = torch.arange(6 * 12, dtype=torch.float32).reshape(6, 12)

    merged = _pad_copy_state_dict(
        {"cross_layer_proj.0.weight": old_weight},
        {"cross_layer_proj.0.weight": torch.zeros(6, 18, dtype=torch.float32)},
        old_state_shape=(2, 2),
        new_state_shape=(3, 2),
        growth_kind="mode",
        grown_axis=0,
    )

    merged_view = merged["cross_layer_proj.0.weight"].reshape(6, 3, 3, 2)
    old_view = old_weight.reshape(6, 3, 2, 2)

    assert torch.equal(merged_view[:, :, :2], old_view)
    assert torch.count_nonzero(merged_view[:, :, 2]) == 0


def test_pad_copy_state_dict_rejects_invalid_state_layout_shape() -> None:
    with pytest.raises(ValueError, match=r"blocks\.0\.mixer\.decay_logit: cannot view old tensor"):
        _pad_copy_state_dict(
            {"blocks.0.mixer.decay_logit": torch.zeros(5, dtype=torch.float32)},
            {"blocks.0.mixer.decay_logit": torch.zeros(3, 3, dtype=torch.float32)},
            old_state_shape=(2, 3),
            new_state_shape=(3, 3),
            growth_kind="mode",
            grown_axis=0,
        )


def test_pad_copy_state_dict_rejects_invalid_reference_slice_shape() -> None:
    with pytest.raises(ValueError, match="residual reference slice"):
        _pad_copy_state_dict(
            {"blocks.0.mixer.decay_logit": torch.ones(2, 3, dtype=torch.float32)},
            {"blocks.0.mixer.decay_logit": torch.zeros(3, 3, dtype=torch.float32)},
            old_state_shape=(2, 3),
            new_state_shape=(3, 3),
            growth_kind="mode",
            grown_axis=0,
            mode_init="residual",
            reference_contexts=[
                {
                    "state_signal": torch.ones(2, 3, 2, dtype=torch.cfloat),
                    "state_signal_features": torch.ones(3, 2, 3, 2),
                    "hidden_features": torch.ones(4),
                }
            ],
        )


def test_mode_prune_copy_state_dict_supports_cross_layer_proj_weights() -> None:
    old_weight = torch.arange(6 * 18, dtype=torch.float32).reshape(6, 18)

    merged = _mode_prune_copy_state_dict(
        {"cross_layer_proj.0.weight": old_weight},
        {"cross_layer_proj.0.weight": torch.zeros(6, 12, dtype=torch.float32)},
        old_state_shape=(3, 2),
        new_state_shape=(2, 2),
        pruned_axis=0,
        pruned_slice=2,
    )

    merged_view = merged["cross_layer_proj.0.weight"].reshape(6, 3, 2, 2)
    old_view = old_weight.reshape(6, 3, 3, 2)

    assert torch.equal(merged_view, old_view[:, :, :2])


def test_try_dynamic_growth_supports_rank_growth() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        num_layers=1,
        dynamic_rank_growth=True,
        max_rank=5,
        growth_check_interval=50,
        residual_saturate_threshold=0.2,
        rank_growth_loss_ceiling=2.5,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))

    new_model, new_config = training._try_dynamic_growth(
        model,
        config,
        dataset,
        torch.device("cpu"),
        recent_losses=[3.0] * 50,
        smoothed_mode_residual_norms=torch.tensor([0.1, 0.1, 0.1]),
    )

    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (2, 3, 4, 2)
    assert new_model.blocks[0].mixer.decay_logit.shape == (2, 3, 4, 2)


def test_try_dynamic_growth_supports_rank_growth_residual_init() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        num_layers=1,
        dynamic_rank_growth=True,
        rank_init="residual",
        max_rank=5,
        growth_check_interval=50,
        residual_saturate_threshold=0.2,
        rank_growth_loss_ceiling=2.5,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    with torch.no_grad():
        model.blocks[0].mixer.decay_logit.copy_(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))
    inputs, _ = sample_causal_lm_batch(dataset.train_tokens, batch_size=4, seq_len=8, device=torch.device("cpu"))

    new_model, new_config = training._try_dynamic_growth(
        model,
        config,
        dataset,
        torch.device("cpu"),
        recent_losses=[3.0] * 50,
        smoothed_mode_residual_norms=torch.tensor([0.1, 0.1, 0.1]),
        reference_token_ids=inputs,
    )

    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (2, 3, 4, 2)
    assert torch.count_nonzero(new_model.blocks[0].mixer.decay_logit[..., 1]) > 0
    assert float(new_model.blocks[0].mixer.decay_logit[..., 1].std().item()) > 0.0


@pytest.mark.parametrize("coupling_type", ["fft", "dwt", "wavelet_packet"])
@pytest.mark.parametrize(
    ("growth_config", "smoothed_residuals", "expected_shape"),
    [
        (
            {"dynamic_mode_growth": True, "max_state_shape": (3, 3, 4), "growth_residual_threshold": 0.2},
            torch.tensor([0.3, 0.1, 0.1]),
            (3, 3, 4),
        ),
        (
            {
                "dynamic_rank_growth": True,
                "max_rank": 5,
                "residual_saturate_threshold": 0.2,
                "rank_growth_loss_ceiling": 2.5,
            },
            torch.tensor([0.1, 0.1, 0.1]),
            (2, 3, 4, 2),
        ),
    ],
)
def test_try_dynamic_growth_supports_spectral_couplings(
    coupling_type: str,
    growth_config: dict,
    smoothed_residuals: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        num_layers=1,
        coupling_type=coupling_type,
        growth_check_interval=50,
        device="cpu",
        seed=0,
        **growth_config,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))

    new_model, new_config = training._try_dynamic_growth(
        model,
        config,
        dataset,
        torch.device("cpu"),
        recent_losses=[3.0] * 50,
        smoothed_mode_residual_norms=smoothed_residuals,
    )

    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == expected_shape
    assert new_model.blocks[0].mixer.decay_logit.shape == expected_shape
    assert not hasattr(new_model.blocks[0].mixer.coupling, "mode_weights")


def test_try_dynamic_growth_uses_mode_growth_reference_contexts_for_residual_init(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        max_state_shape=(3, 3, 4),
        num_layers=1,
        dynamic_mode_growth=True,
        mode_init="residual",
        growth_check_interval=50,
        growth_residual_threshold=0.2,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    inputs, _ = sample_causal_lm_batch(dataset.train_tokens, batch_size=4, seq_len=8, device=torch.device("cpu"))
    calls = {"count": 0}

    def fake_mode_reference_contexts(model, token_ids):
        calls["count"] += 1
        state_signal = torch.ones(2, 3, 4, dtype=torch.cfloat)
        return [
            {
                "hidden_features": torch.ones(model.hidden_size * 2),
                "state_signal": state_signal,
                "state_signal_features": training._state_signal_features(state_signal),
            }
        ]

    def fail_rank_reference_contexts(*args, **kwargs):
        raise AssertionError("mode residual init should not use rank-growth reference contexts")

    monkeypatch.setattr(training, "_collect_growth_reference_contexts", fake_mode_reference_contexts)
    monkeypatch.setattr(training, "_collect_rank_growth_reference_contexts", fail_rank_reference_contexts)

    new_model, new_config = training._try_dynamic_growth(
        model,
        config,
        dataset,
        torch.device("cpu"),
        recent_losses=[3.0] * 50,
        smoothed_mode_residual_norms=torch.tensor([0.3, 0.1, 0.1]),
        reference_token_ids=inputs,
    )

    assert calls["count"] == 1
    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (3, 3, 4)


def test_try_dynamic_growth_uses_rank_residual_reference_contexts(monkeypatch) -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        num_layers=1,
        dynamic_rank_growth=True,
        rank_init="residual",
        max_rank=5,
        growth_check_interval=50,
        residual_saturate_threshold=0.2,
        rank_growth_loss_ceiling=2.5,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    inputs, _ = sample_causal_lm_batch(dataset.train_tokens, batch_size=4, seq_len=8, device=torch.device("cpu"))
    calls = {"count": 0}

    def fake_rank_reference_contexts(model, token_ids):
        calls["count"] += 1
        residual_signal = torch.ones(2, 3, 4, dtype=torch.cfloat)
        return [
            {
                "hidden_residual_features": torch.ones(model.hidden_size * 2),
                "rank_residual_state_signal": residual_signal,
                "rank_residual_state_features": training._state_signal_features(residual_signal),
            }
        ]

    def fail_mode_reference_contexts(*args, **kwargs):
        raise AssertionError("rank residual init should not use mode-growth reference contexts")

    monkeypatch.setattr(training, "_collect_rank_growth_reference_contexts", fake_rank_reference_contexts)
    monkeypatch.setattr(training, "_collect_growth_reference_contexts", fail_mode_reference_contexts)

    new_model, new_config = training._try_dynamic_growth(
        model,
        config,
        dataset,
        torch.device("cpu"),
        recent_losses=[3.0] * 50,
        smoothed_mode_residual_norms=torch.tensor([0.1, 0.1, 0.1]),
        reference_token_ids=inputs,
    )

    assert calls["count"] == 1
    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (2, 3, 4, 2)


def test_try_dynamic_growth_supports_mode_growth_mean_init() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        max_state_shape=(3, 3, 4),
        num_layers=1,
        dynamic_mode_growth=True,
        mode_init="mean",
        growth_check_interval=50,
        growth_residual_threshold=0.2,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    with torch.no_grad():
        model.blocks[0].mixer.decay_logit.copy_(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))

    new_model, new_config = training._try_dynamic_growth(
        model,
        config,
        dataset,
        torch.device("cpu"),
        recent_losses=[3.0] * 50,
        smoothed_mode_residual_norms=torch.tensor([0.3, 0.1, 0.1]),
    )

    assert new_model is not None
    assert new_config is not None
    expected = model.blocks[0].mixer.decay_logit.mean(dim=0)
    assert new_config.state_shape == (3, 3, 4)
    assert torch.allclose(new_model.blocks[0].mixer.decay_logit[2], expected)


def test_try_dynamic_growth_supports_rank_growth_mean_init() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        num_layers=1,
        dynamic_rank_growth=True,
        rank_init="mean",
        max_rank=5,
        growth_check_interval=50,
        growth_residual_threshold=0.2,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    with torch.no_grad():
        model.blocks[0].mixer.decay_logit.copy_(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))

    new_model, new_config = training._try_dynamic_growth(
        model,
        config,
        dataset,
        torch.device("cpu"),
        recent_losses=[3.0] * 50,
        smoothed_mode_residual_norms=torch.tensor([0.1, 0.1, 0.1]),
    )

    assert new_model is not None
    assert new_config is not None
    expected = model.blocks[0].mixer.decay_logit.mean()
    assert new_config.state_shape == (2, 3, 4, 2)
    assert torch.allclose(
        new_model.blocks[0].mixer.decay_logit[..., 1],
        torch.full((2, 3, 4), expected.item()),
    )


def test_try_dynamic_growth_supports_mode_growth_orthogonal_init() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        max_state_shape=(3, 3, 4),
        num_layers=1,
        dynamic_mode_growth=True,
        mode_init="orthogonal",
        growth_check_interval=50,
        growth_residual_threshold=0.2,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    inputs, _ = sample_causal_lm_batch(dataset.train_tokens, batch_size=4, seq_len=8, device=torch.device("cpu"))

    new_model, new_config = training._try_dynamic_growth(
        model,
        config,
        dataset,
        torch.device("cpu"),
        recent_losses=[3.0] * 50,
        smoothed_mode_residual_norms=torch.tensor([0.3, 0.1, 0.1]),
        reference_token_ids=inputs,
    )

    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (3, 3, 4)
    new_rows = new_model.blocks[0].mixer.signal_projector.magnitude_proj.weight.reshape(3, 3, 4, -1)[2]
    assert torch.count_nonzero(new_rows) > 0


def test_select_dynamic_growth_action_prefers_mode_growth_when_residual_threshold_is_exceeded() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4),
        max_state_shape=(3, 4, 4),
        dynamic_mode_growth=True,
        dynamic_rank_growth=True,
        max_rank=5,
        growth_check_interval=50,
        growth_residual_threshold=0.25,
    )

    action = training._select_dynamic_growth_action(
        config,
        smoothed_mode_residual_norms=torch.tensor([0.3, 0.2, 0.1]),
        recent_losses=[3.0] * 50,
    )

    assert action == ("mode", 0)


def test_select_dynamic_growth_action_respects_effective_cooldown_threshold() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4),
        max_state_shape=(3, 4, 4),
        dynamic_mode_growth=True,
        growth_residual_threshold=0.25,
    )

    action = training._select_dynamic_growth_action(
        config,
        smoothed_mode_residual_norms=torch.tensor([0.3, 0.2, 0.1]),
        recent_losses=[3.0] * 50,
        effective_growth_residual_threshold=0.35,
    )

    assert action is None


def test_select_dynamic_growth_action_uses_rank_growth_when_mode_residuals_are_saturated_and_loss_is_high() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4),
        dynamic_mode_growth=False,
        dynamic_rank_growth=True,
        max_rank=5,
        growth_check_interval=50,
        growth_residual_threshold=0.25,
        residual_saturate_threshold=0.15,
        rank_growth_loss_ceiling=2.5,
    )

    action = training._select_dynamic_growth_action(
        config,
        smoothed_mode_residual_norms=torch.tensor([0.1, 0.12, 0.15]),
        recent_losses=[3.0] * 50,
    )

    assert action == ("rank", 3)


def test_select_dynamic_growth_action_blocks_rank_growth_when_any_mode_residual_is_above_saturate_threshold() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4),
        dynamic_mode_growth=False,
        dynamic_rank_growth=True,
        max_rank=5,
        growth_check_interval=50,
        growth_residual_threshold=0.25,
        residual_saturate_threshold=0.15,
        rank_growth_loss_ceiling=2.5,
    )

    action = training._select_dynamic_growth_action(
        config,
        smoothed_mode_residual_norms=torch.tensor([0.1, 0.2, 0.15]),
        recent_losses=[3.0] * 50,
    )

    assert action is None


def test_select_dynamic_growth_action_blocks_rank_growth_when_recent_loss_is_below_ceiling() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4),
        dynamic_mode_growth=False,
        dynamic_rank_growth=True,
        max_rank=5,
        growth_check_interval=50,
        growth_residual_threshold=0.25,
        residual_saturate_threshold=0.15,
        rank_growth_loss_ceiling=2.5,
    )

    action = training._select_dynamic_growth_action(
        config,
        smoothed_mode_residual_norms=torch.tensor([0.1, 0.12, 0.15]),
        recent_losses=[2.0] * 50,
    )

    assert action is None


def test_try_dynamic_mode_pruning_shrinks_selected_mode_dimension() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        max_state_shape=(3, 5, 6),
        num_layers=1,
        dynamic_mode_pruning=True,
        growth_check_interval=50,
        prune_threshold=0.2,
        prune_sustain_steps=1,
        prune_min_steps=0,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    with torch.no_grad():
        model.blocks[0].mixer.decay_logit.copy_(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))

    new_model, new_config = training._try_dynamic_mode_pruning(
        model,
        config,
        dataset,
        torch.device("cpu"),
        smoothed_mode_pruning_residual_norms=torch.tensor([0.3, 0.1, 0.4]),
        smoothed_mode_slice_activation_variances=[
            torch.tensor([0.3, 0.2]),
            torch.tensor([0.4, 0.1, 0.3]),
            torch.tensor([0.2, 0.2, 0.2, 0.2]),
        ],
        prune_candidate_streaks=[0, 1, 0],
        mode_last_growth_steps=[0, 0, 0],
        axis_kinds=["mode", "mode", "mode"],
        step=50,
    )

    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (2, 2, 4)
    assert new_config.max_state_shape == (3, 5, 6)
    assert torch.allclose(
        new_model.blocks[0].mixer.decay_logit,
        torch.stack(
            [
                model.blocks[0].mixer.decay_logit[:, 0, :],
                model.blocks[0].mixer.decay_logit[:, 2, :],
            ],
            dim=1,
        ),
    )
    assert torch.allclose(
        new_model.blocks[0].mixer.coupling.mode_weights[1],
        model.blocks[0].mixer.coupling.mode_weights[1].index_select(0, torch.tensor([0, 2])).index_select(
            1, torch.tensor([0, 2])
        ),
    )


@pytest.mark.parametrize("coupling_type", ["fft", "dwt", "wavelet_packet"])
def test_try_dynamic_mode_pruning_supports_spectral_couplings(coupling_type: str) -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4),
        max_state_shape=(3, 5, 6),
        num_layers=1,
        coupling_type=coupling_type,
        dynamic_mode_pruning=True,
        growth_check_interval=50,
        prune_threshold=0.2,
        prune_sustain_steps=1,
        prune_min_steps=0,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    with torch.no_grad():
        model.blocks[0].mixer.decay_logit.copy_(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))

    new_model, new_config = training._try_dynamic_mode_pruning(
        model,
        config,
        dataset,
        torch.device("cpu"),
        smoothed_mode_pruning_residual_norms=torch.tensor([0.3, 0.1, 0.4]),
        smoothed_mode_slice_activation_variances=[
            torch.tensor([0.3, 0.2]),
            torch.tensor([0.4, 0.1, 0.3]),
            torch.tensor([0.2, 0.2, 0.2, 0.2]),
        ],
        prune_candidate_streaks=[0, 1, 0],
        mode_last_growth_steps=[0, 0, 0],
        axis_kinds=["mode", "mode", "mode"],
        step=50,
    )

    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (2, 2, 4)
    assert torch.allclose(
        new_model.blocks[0].mixer.decay_logit,
        torch.stack(
            [
                model.blocks[0].mixer.decay_logit[:, 0, :],
                model.blocks[0].mixer.decay_logit[:, 2, :],
            ],
            dim=1,
        ),
    )
    assert not hasattr(new_model.blocks[0].mixer.coupling, "mode_weights")


def test_select_dynamic_mode_pruning_action_respects_min_steps() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4),
        dynamic_mode_pruning=True,
        prune_threshold=0.25,
        prune_sustain_steps=1,
        prune_min_steps=5,
    )

    action = training._select_dynamic_mode_pruning_action(
        config,
        smoothed_mode_pruning_residual_norms=torch.tensor([0.3, 0.1, 0.3]),
        prune_candidate_streaks=[0, 1, 0],
        mode_last_growth_steps=[0, 49, 0],
        axis_kinds=["mode", "mode", "mode"],
        step=50,
    )

    assert action is None


def test_select_dynamic_mode_pruning_action_requires_sustain_steps() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4),
        dynamic_mode_pruning=True,
        prune_threshold=0.25,
        prune_sustain_steps=2,
        prune_min_steps=0,
    )

    action = training._select_dynamic_mode_pruning_action(
        config,
        smoothed_mode_pruning_residual_norms=torch.tensor([0.3, 0.1, 0.3]),
        prune_candidate_streaks=[0, 1, 0],
        mode_last_growth_steps=[0, 0, 0],
        axis_kinds=["mode", "mode", "mode"],
        step=50,
    )

    assert action is None


def test_select_dynamic_mode_pruning_action_uses_prune_threshold_not_growth_threshold() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4),
        dynamic_mode_pruning=True,
        growth_residual_threshold=0.01,
        prune_threshold=0.25,
        prune_sustain_steps=1,
        prune_min_steps=0,
    )

    action = training._select_dynamic_mode_pruning_action(
        config,
        smoothed_mode_pruning_residual_norms=torch.tensor([0.3, 0.2, 0.3]),
        prune_candidate_streaks=[0, 1, 0],
        mode_last_growth_steps=[0, 0, 0],
        axis_kinds=["mode", "mode", "mode"],
        step=50,
    )

    assert action == 1


def test_select_mode_slice_to_prune_uses_lowest_activation_variance() -> None:
    config = TrainingConfig(state_shape=(2, 3, 4))

    pruned_slice = training._select_mode_slice_to_prune(
        config,
        [
            torch.tensor([0.4, 0.1]),
            torch.tensor([0.3, 0.05, 0.2]),
            torch.tensor([0.6, 0.7, 0.8, 0.9]),
        ],
        pruned_axis=1,
    )

    assert pruned_slice == 1


def test_select_dynamic_rank_pruning_action_only_prunes_rank_axes() -> None:
    config = TrainingConfig(
        state_shape=(2, 3, 4, 2),
        dynamic_rank_pruning=True,
        prune_threshold=0.25,
        prune_sustain_steps=2,
        prune_min_steps=5,
    )

    action = training._select_dynamic_rank_pruning_action(
        config,
        smoothed_mode_pruning_residual_norms=torch.tensor([0.05, 0.1, 0.15, 0.2]),
        prune_candidate_streaks=[2, 2, 2, 2],
        mode_last_growth_steps=[0, 0, 0, 10],
        axis_kinds=["mode", "mode", "mode", "rank"],
        step=20,
    )

    assert action == 3


def test_try_dynamic_rank_pruning_supports_rank_axis_prune() -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4, 2),
        max_state_shape=(3, 5, 6, 2),
        num_layers=1,
        dynamic_rank_pruning=True,
        growth_check_interval=50,
        prune_threshold=0.2,
        prune_sustain_steps=1,
        prune_min_steps=0,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    with torch.no_grad():
        model.blocks[0].mixer.decay_logit.copy_(torch.arange(48, dtype=torch.float32).reshape(2, 3, 4, 2))

    new_model, new_config = training._try_dynamic_rank_pruning(
        model,
        config,
        dataset,
        torch.device("cpu"),
        smoothed_mode_pruning_residual_norms=torch.tensor([0.3, 0.4, 0.5, 0.1]),
        prune_candidate_streaks=[0, 0, 0, 1],
        mode_last_growth_steps=[0, 0, 0, 0],
        axis_kinds=["mode", "mode", "mode", "rank"],
        step=50,
    )

    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (2, 3, 4)
    assert new_config.max_state_shape == (3, 5, 6)
    assert torch.allclose(
        new_model.blocks[0].mixer.decay_logit,
        model.blocks[0].mixer.decay_logit.mean(dim=3),
    )


@pytest.mark.parametrize("coupling_type", ["fft", "dwt", "wavelet_packet"])
def test_try_dynamic_rank_pruning_supports_spectral_couplings(coupling_type: str) -> None:
    dataset = build_text_dataset("abcde " * 40, val_fraction=0.2, min_split_tokens=16)
    config = TrainingConfig(
        hidden_size=12,
        state_shape=(2, 3, 4, 2),
        max_state_shape=(3, 5, 6, 2),
        num_layers=1,
        coupling_type=coupling_type,
        dynamic_rank_pruning=True,
        growth_check_interval=50,
        prune_threshold=0.2,
        prune_sustain_steps=1,
        prune_min_steps=0,
        device="cpu",
        seed=0,
    )
    model = training._build_model_from_config(config, dataset, torch.device("cpu"))
    with torch.no_grad():
        model.blocks[0].mixer.decay_logit.copy_(torch.arange(48, dtype=torch.float32).reshape(2, 3, 4, 2))

    new_model, new_config = training._try_dynamic_rank_pruning(
        model,
        config,
        dataset,
        torch.device("cpu"),
        smoothed_mode_pruning_residual_norms=torch.tensor([0.3, 0.4, 0.5, 0.1]),
        prune_candidate_streaks=[0, 0, 0, 1],
        mode_last_growth_steps=[0, 0, 0, 0],
        axis_kinds=["mode", "mode", "mode", "rank"],
        step=50,
    )

    assert new_model is not None
    assert new_config is not None
    assert new_config.state_shape == (2, 3, 4)
    assert torch.allclose(
        new_model.blocks[0].mixer.decay_logit,
        model.blocks[0].mixer.decay_logit.mean(dim=3),
    )
    assert not hasattr(new_model.blocks[0].mixer.coupling, "mode_weights")


def test_update_pruning_candidate_streaks_accumulates_below_threshold_checks() -> None:
    streaks = training._update_pruning_candidate_streaks(
        [0, 1, 2],
        torch.tensor([0.1, 0.5, 0.2]),
        threshold=0.25,
    )

    assert streaks == [1, 0, 3]


def test_update_mode_last_growth_steps_tracks_growth_and_prune_events() -> None:
    assert training._update_mode_last_growth_steps([0, 0], (2, 3), (2, 4), step=10) == [0, 10]
    assert training._update_mode_last_growth_steps([0, 10], (2, 4), (2, 4, 2), step=20) == [0, 10, 20]
    assert training._update_mode_last_growth_steps([0, 10], (2, 4), (2, 3), step=30) == [0, 10]
    assert training._update_mode_last_growth_steps([0, 10, 20], (2, 4, 2), (2, 4), step=30) == [0, 10]


def test_update_state_axis_kinds_tracks_rank_growth_and_prune_events() -> None:
    assert training._update_state_axis_kinds(["mode", "mode"], (2, 3), (2, 4)) == ["mode", "mode"]
    assert training._update_state_axis_kinds(["mode", "mode"], (2, 4), (2, 3)) == ["mode", "mode"]
    assert training._update_state_axis_kinds(["mode", "mode"], (2, 4), (2, 4, 2)) == ["mode", "mode", "rank"]
    assert training._update_state_axis_kinds(["mode", "mode", "rank"], (2, 4, 2), (2, 4)) == ["mode", "mode"]
