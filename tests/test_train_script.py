import importlib.util
from pathlib import Path


def load_train_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_parser_exposes_phase_and_growth_defaults() -> None:
    train_script = load_train_script()

    args = train_script.build_parser().parse_args([])

    assert args.corpus == "greek_classics"
    assert args.max_chars == 100000
    assert args.batch_size == 8
    assert args.seq_len == 128
    assert args.steps == 2000
    assert args.eval_every == 100
    assert args.eval_batches == 4
    assert args.lr == 1e-3
    assert args.hidden_size == 256
    assert args.state_shape == (4, 4, 4)
    assert args.token_magnitude_type == "inverse_frequency_learned"
    assert args.token_phase == "semantic"
    assert args.readout_type == "phase_aware"
    assert args.coupling_type == "sequential"
    assert args.enable_self_relation is True
    assert args.low_frequency_gain == 0.5
    assert args.low_frequency_sigma == 0.35
    assert args.high_frequency_gain == 0.5
    assert args.high_frequency_cutoff == 0.5
    assert args.dynamic_spectral_gains is False
    assert args.anisotropic_spectral_gains is False
    assert args.wavelet_levels is None
    assert args.chunk_size is None
    assert args.track_chunk_drift is False
    assert args.attention_every_k == 0
    assert args.attention_num_heads == 8
    assert args.attention_window == 256
    assert args.attention_position == "after"
    assert args.stateful_training is True
    assert args.lr_warmup_steps == 0
    assert args.lr_decay_style == "constant"
    assert args.min_lr_scale == 0.1
    assert args.grad_clip_norm is None
    assert args.weight_decay == 0.0
    assert args.normalization_type == "frobenius"
    assert args.enable_cross_layer_state is False
    assert args.dynamic_mode_growth is False
    assert args.dynamic_rank_growth is False
    assert args.dynamic_mode_pruning is False
    assert args.dynamic_rank_pruning is False
    assert args.max_rank is None
    assert args.max_state_shape is None
    assert args.growth_check_interval == 50
    assert args.growth_residual_threshold == 0.4
    assert args.post_growth_cooldown_checks == 0
    assert args.post_growth_cooldown_threshold_scale == 1.5
    assert args.residual_saturate_threshold == 0.4
    assert args.growth_residual_ema_decay == 0.95
    assert args.record_residual_diagnostics is False
    assert args.diagnostics_out is None
    assert args.min_checks_before_first_growth == 0
    assert args.rank_growth_loss_ceiling == 1.5
    assert args.prune_threshold == 0.4
    assert args.prune_sustain_steps == 1
    assert args.prune_min_steps == 50
    assert args.mode_init == "zero"
    assert args.rank_init == "zero"
    assert args.generation_eval_samples == 0
    assert args.generation_prompt_len == 64
    assert args.generation_new_tokens == 128
    assert args.generation_temperature == 0.8
    assert args.generation_top_k == 20
    assert args.benchmark_prompt_lengths == ()
    assert args.benchmark_new_tokens == 128
    assert args.resume_from is None


def test_build_parser_parses_max_state_shape_like_state_shape() -> None:
    train_script = load_train_script()

    args = train_script.build_parser().parse_args(
        [
            "--normalization-type",
            "per_mode",
            "--token-phase",
            "semantic_virtual_offset",
            "--enable-cross-layer-state",
            "--disable-self-relation",
            "--coupling-type",
            "wavelet-packet-phase",
            "--low-frequency-gain",
            "0.8",
            "--low-frequency-sigma",
            "0.2",
            "--high-frequency-gain",
            "0.4",
            "--high-frequency-cutoff",
            "0.65",
            "--dynamic-spectral-gains",
            "--anisotropic-spectral-gains",
            "--wavelet-levels",
            "3",
            "--chunk-size",
            "16",
            "--track-chunk-drift",
            "--dynamic-mode-growth",
            "--dynamic-rank-growth",
            "--dynamic-mode-pruning",
            "--dynamic-rank-pruning",
            "--growth-residual-threshold",
            "0.35",
            "--post-growth-cooldown-checks",
            "3",
            "--post-growth-cooldown-threshold-scale",
            "1.8",
            "--residual-saturate-threshold",
            "0.12",
            "--growth-residual-ema-decay",
            "0.9",
            "--lr-warmup-steps",
            "25",
            "--lr-decay-style",
            "cosine",
            "--min-lr-scale",
            "0.2",
            "--grad-clip-norm",
            "1.5",
            "--weight-decay",
            "0.03",
            "--record-residual-diagnostics",
            "--diagnostics-out",
            "runs/diag.json",
            "--min-checks-before-first-growth",
            "3",
            "--rank-growth-loss-ceiling",
            "2.75",
            "--prune-threshold",
            "0.08",
            "--prune-sustain-steps",
            "3",
            "--prune-min-steps",
            "9",
            "--max-rank",
            "6",
            "--max-state-shape",
            "3,4,5",
            "--growth-check-interval",
            "75",
            "--mode-init",
            "residual",
            "--rank-init",
            "residual",
            "--generation-eval-samples",
            "4",
            "--generation-prompt-len",
            "48",
            "--generation-new-tokens",
            "96",
            "--generation-temperature",
            "0.7",
            "--generation-top-k",
            "12",
            "--benchmark-prompt-lengths",
            "16,64,256",
            "--benchmark-new-tokens",
            "192",
        ]
    )

    assert args.normalization_type == "per_mode"
    assert args.token_phase == "semantic_virtual_offset"
    assert args.enable_cross_layer_state is True
    assert args.enable_self_relation is False
    assert args.coupling_type == "wavelet_packet_max_gauge"
    assert args.low_frequency_gain == 0.8
    assert args.low_frequency_sigma == 0.2
    assert args.high_frequency_gain == 0.4
    assert args.high_frequency_cutoff == 0.65
    assert args.dynamic_spectral_gains is True
    assert args.anisotropic_spectral_gains is True
    assert args.wavelet_levels == 3
    assert args.chunk_size == 16
    assert args.track_chunk_drift is True
    assert args.dynamic_mode_growth is True
    assert args.dynamic_rank_growth is True
    assert args.dynamic_mode_pruning is True
    assert args.dynamic_rank_pruning is True
    assert args.growth_residual_threshold == 0.35
    assert args.post_growth_cooldown_checks == 3
    assert args.post_growth_cooldown_threshold_scale == 1.8
    assert args.residual_saturate_threshold == 0.12
    assert args.growth_residual_ema_decay == 0.9
    assert args.lr_warmup_steps == 25
    assert args.lr_decay_style == "cosine"
    assert args.min_lr_scale == 0.2
    assert args.grad_clip_norm == 1.5
    assert args.weight_decay == 0.03
    assert args.record_residual_diagnostics is True
    assert args.diagnostics_out == "runs/diag.json"
    assert args.rank_growth_loss_ceiling == 2.75
    assert args.prune_threshold == 0.08
    assert args.prune_sustain_steps == 3
    assert args.prune_min_steps == 9
    assert args.max_rank == 6
    assert args.max_state_shape == (3, 4, 5)
    assert args.growth_check_interval == 75
    assert args.min_checks_before_first_growth == 3
    assert args.mode_init == "residual"
    assert args.rank_init == "residual"
    assert args.generation_eval_samples == 4
    assert args.generation_prompt_len == 48
    assert args.generation_new_tokens == 96
    assert args.generation_temperature == 0.7
    assert args.generation_top_k == 12
    assert args.benchmark_prompt_lengths == (16, 64, 256)
    assert args.benchmark_new_tokens == 192


def test_build_parser_parses_hybrid_attention_and_stateless_flags() -> None:
    train_script = load_train_script()

    args = train_script.build_parser().parse_args(
        [
            "--attention-every-k",
            "3",
            "--attention-num-heads",
            "4",
            "--attention-window",
            "128",
            "--attention-position",
            "before",
            "--stateless-training",
        ]
    )

    assert args.attention_every_k == 3
    assert args.attention_num_heads == 4
    assert args.attention_window == 128
    assert args.attention_position == "before"
    assert args.stateful_training is False


def test_resume_helpers_restore_checkpoint_config_and_metrics() -> None:
    train_script = load_train_script()
    payload = {
        "step": 7,
        "model_state_dict": {"weight": object()},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "config": {
            "steps": 500,
            "device": "cuda",
            "state_shape": [2, 3, 4],
            "max_state_shape": [3, 4, 5],
            "benchmark_prompt_lengths": [16, 64],
        },
        "train_losses": [1.25, 1.0],
        "val_losses": [[1, 1.5]],
        "train_metrics": [
            {"step": 1, "loss": 1.25, "accuracy": 0.2, "perplexity": 3.5, "bpc": 1.8},
        ],
        "val_metrics": [
            {"step": 1, "loss": 1.5, "accuracy": 0.1, "perplexity": 4.5, "bpc": 2.1},
        ],
    }

    config = train_script.training_config_from_checkpoint(payload)
    args = train_script.build_parser().parse_args(
        ["--resume-from", "checkpoint.pt", "--steps", "750", "--device", "cpu"]
    )
    config = train_script._apply_resume_overrides(
        config,
        args,
        ["--resume-from", "checkpoint.pt", "--steps", "750", "--device", "cpu"],
    )
    resume_state = train_script.resume_state_from_checkpoint(payload)

    assert config.steps == 750
    assert config.device == "cpu"
    assert config.state_shape == (2, 3, 4)
    assert config.max_state_shape == (3, 4, 5)
    assert config.benchmark_prompt_lengths == (16, 64)
    assert resume_state.step == 7
    assert resume_state.optimizer_state_dict is payload["optimizer_state_dict"]
    assert resume_state.train_losses == [1.25, 1.0]
    assert resume_state.val_losses == [(1, 1.5)]
    assert resume_state.train_metrics[0][0] == 1
    assert resume_state.train_metrics[0][1].bpc == 1.8
    assert resume_state.val_metrics[0][1].accuracy == 0.1
