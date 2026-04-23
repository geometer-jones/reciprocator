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

    assert args.token_magnitude_type == "inverse_frequency_learned"
    assert args.token_phase == "semantic"
    assert args.readout_type == "phase_aware"
    assert args.coupling_type == "sequential"
    assert args.low_frequency_gain == 0.5
    assert args.low_frequency_sigma == 0.35
    assert args.high_frequency_gain == 0.5
    assert args.high_frequency_cutoff == 0.5
    assert args.dynamic_spectral_gains is False
    assert args.anisotropic_spectral_gains is False
    assert args.wavelet_levels is None
    assert args.chunk_size is None
    assert args.track_chunk_drift is False
    assert args.lr_warmup_steps == 0
    assert args.lr_decay_style == "constant"
    assert args.min_lr_scale == 0.1
    assert args.grad_clip_norm is None
    assert args.weight_decay == 0.0
    assert args.normalization_type == "frobenius"
    assert args.enable_anticipator_relation is False
    assert args.enable_cross_layer_state is False
    assert args.dynamic_mode_growth is False
    assert args.dynamic_rank_growth is False
    assert args.dynamic_mode_pruning is False
    assert args.dynamic_rank_pruning is False
    assert args.max_rank is None
    assert args.max_state_shape is None
    assert args.growth_check_interval == 50
    assert args.growth_residual_threshold == 0.4
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


def test_build_parser_parses_max_state_shape_like_state_shape() -> None:
    train_script = load_train_script()

    args = train_script.build_parser().parse_args(
        [
            "--normalization-type",
            "per_mode",
            "--token-phase",
            "semantic_virtual_offset",
            "--enable-anticipator-relation",
            "--enable-cross-layer-state",
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
    assert args.enable_anticipator_relation is True
    assert args.enable_cross_layer_state is True
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
