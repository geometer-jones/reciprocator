#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reciprocator import available_corpora
from reciprocator.mixer import canonicalize_coupling_type
from reciprocator.training import TrainingConfig, train_model


def parse_state_shape(raw: str) -> Tuple[int, ...]:
    pieces = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    if not pieces:
        raise argparse.ArgumentTypeError("state_shape must contain at least one dimension.")
    if any(piece <= 0 for piece in pieces):
        raise argparse.ArgumentTypeError("state_shape dimensions must be positive integers.")
    return pieces


def parse_positive_int_list(raw: str) -> Tuple[int, ...]:
    pieces = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    if any(piece <= 0 for piece in pieces):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")
    return pieces


def parse_normalization_type(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in {"frobenius", "per_mode"}:
        raise argparse.ArgumentTypeError("normalization_type must be one of {'frobenius', 'per_mode'}.")
    return normalized


def parse_mode_init(raw: str) -> str:
    normalized = raw.strip().lower().replace("-", "_")
    if normalized not in {"zero", "mean", "orthogonal", "residual"}:
        raise argparse.ArgumentTypeError("mode_init must be one of {'zero', 'mean', 'orthogonal', 'residual'}.")
    return normalized


def parse_rank_init(raw: str) -> str:
    normalized = raw.strip().lower().replace("-", "_")
    if normalized not in {"zero", "mean", "residual"}:
        raise argparse.ArgumentTypeError("rank_init must be one of {'zero', 'mean', 'residual'}.")
    return normalized


def parse_coupling_type(raw: str) -> str:
    try:
        return canonicalize_coupling_type(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    corpora = [corpus.name for corpus in available_corpora()]
    parser = argparse.ArgumentParser(description="Run a character-level training loop.")
    parser.add_argument("--corpus", choices=corpora, default="greek_classics")
    parser.add_argument("--max-chars", type=int, default=100000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--lr-decay-style",
        choices=("constant", "linear", "cosine"),
        default="constant",
        help="Learning-rate decay applied after warmup.",
    )
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--state-shape", type=parse_state_shape, default=(4, 4, 4))
    parser.add_argument(
        "--max-state-shape",
        type=parse_state_shape,
        default=None,
        help="Optional max growth target, parsed like --state-shape (for example: 2,3,4).",
    )
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ffn-expansion-factor", type=int, default=2)
    parser.add_argument("--readout-type", choices=("magnitude", "phase_aware"), default="phase_aware")
    parser.add_argument(
        "--token-magnitude-type",
        choices=("learned", "inverse_frequency", "inverse_frequency_learned"),
        default="inverse_frequency_learned",
    )
    parser.add_argument("--phase-type", choices=("rope", "locked_wave", "local_wave"), default="rope")
    parser.add_argument(
        "--token-phase",
        choices=("none", "semantic", "virtual_offset", "semantic_virtual_offset"),
        default="semantic",
    )
    parser.add_argument(
        "--normalization-type",
        type=parse_normalization_type,
        default="frobenius",
        help="Tensor-state normalization: frobenius or per_mode.",
    )
    self_relation_group = parser.add_mutually_exclusive_group()
    self_relation_group.add_argument(
        "--enable-self-relation",
        dest="enable_self_relation",
        action="store_true",
        default=True,
        help="Add the optional prior-state × tentative-present Hadamard self-relation term (enabled by default).",
    )
    self_relation_group.add_argument(
        "--disable-self-relation",
        dest="enable_self_relation",
        action="store_false",
        help="Disable the prior-state × tentative-present Hadamard self-relation term.",
    )
    parser.add_argument(
        "--dynamic-spectral-gains",
        action="store_true",
        default=False,
        help="Use a zero-initialized signal-conditioned projector for spectral coupling gains.",
    )
    parser.add_argument(
        "--anisotropic-spectral-gains",
        action="store_true",
        default=False,
        help="Use full coordinatewise FFT dynamic spectral gains instead of radial sampled gains.",
    )
    parser.add_argument(
        "--enable-anticipator-relation",
        action="store_true",
        help="Use the previous step's model output as a next-step prediction relation.",
    )
    parser.add_argument(
        "--enable-cross-layer-state",
        action="store_true",
        help="Inject a read-only hidden correction derived from the previous Reciprocator layer state.",
    )
    parser.add_argument(
        "--coupling-type",
        type=parse_coupling_type,
        default="sequential",
        help="Recurrent tensor coupling backend: sequential, fft, dwt, wavelet_packet, or wavelet_packet_max_gauge.",
    )
    parser.add_argument("--low-frequency-gain", type=float, default=0.5)
    parser.add_argument("--low-frequency-sigma", type=float, default=0.35)
    parser.add_argument("--high-frequency-gain", type=float, default=0.5)
    parser.add_argument("--high-frequency-cutoff", type=float, default=0.5)
    parser.add_argument(
        "--wavelet-levels",
        type=int,
        default=None,
        help="Optional max number of 1-D Haar wavelet levels for dwt and wavelet-packet couplings.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional training-time chunk size for the mixer forward path. None keeps exact sequential execution.",
    )
    parser.add_argument(
        "--track-chunk-drift",
        action="store_true",
        default=False,
        help="Track state drift within chunked mixer updates during training.",
    )
    parser.add_argument(
        "--attention-every-k",
        type=int,
        default=0,
        help="Insert a LocalAttentionBlock every k Reciprocator blocks. Use 0 to disable hybrid attention.",
    )
    parser.add_argument(
        "--attention-num-heads",
        type=int,
        default=8,
        help="Number of local-attention heads when hybrid attention is enabled.",
    )
    parser.add_argument(
        "--attention-window",
        type=int,
        default=256,
        help="Sliding local-attention window size when hybrid attention is enabled.",
    )
    parser.add_argument(
        "--attention-position",
        choices=("before", "after"),
        default="after",
        help="Whether local attention is inserted before or after each Reciprocator group.",
    )
    stateful_group = parser.add_mutually_exclusive_group()
    stateful_group.add_argument(
        "--stateful-training",
        dest="stateful_training",
        action="store_true",
        default=True,
        help="Carry recurrent state forward across batches during training.",
    )
    stateful_group.add_argument(
        "--stateless-training",
        dest="stateful_training",
        action="store_false",
        help="Reset recurrent state each batch during training.",
    )
    parser.add_argument("--device", default="cpu", help="One of cpu, cuda, mps, or auto.")
    parser.add_argument(
        "--no-tensor-dynamic-growth",
        action="store_false",
        dest="tensor_dynamic_growth",
        default=True,
        help="Disable CUDA expandable-segment allocator growth.",
    )
    parser.add_argument("--dynamic-mode-growth", action="store_true", default=False)
    parser.add_argument("--dynamic-rank-growth", action="store_true", default=False)
    parser.add_argument("--dynamic-mode-pruning", action="store_true", default=False)
    parser.add_argument("--dynamic-rank-pruning", action="store_true", default=False)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument("--growth-check-interval", type=int, default=50)
    parser.add_argument("--growth-residual-threshold", type=float, default=0.4)
    parser.add_argument(
        "--post-growth-cooldown-checks",
        type=int,
        default=0,
        help="Number of growth-check intervals after a growth event that use an elevated growth threshold.",
    )
    parser.add_argument(
        "--post-growth-cooldown-threshold-scale",
        type=float,
        default=1.5,
        help="Multiplier applied to growth_residual_threshold during post-growth cooldown checks.",
    )
    parser.add_argument("--residual-saturate-threshold", type=float, default=0.4)
    parser.add_argument("--growth-residual-ema-decay", type=float, default=0.95)
    parser.add_argument(
        "--record-residual-diagnostics",
        action="store_true",
        default=False,
        help="Record residual and redundancy EMA diagnostics at each growth check.",
    )
    parser.add_argument(
        "--diagnostics-out",
        type=str,
        default=None,
        help="Optional path for residual diagnostics JSON output.",
    )
    parser.add_argument(
        "--min-checks-before-first-growth",
        type=int,
        default=0,
        help="Minimum number of elapsed growth intervals before dynamic growth becomes eligible.",
    )
    parser.add_argument(
        "--rank-growth-loss-ceiling",
        type=float,
        default=1.5,
        help="Recent loss must stay above this threshold before saturated modes can trigger rank growth.",
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.4,
        help="Redundancy EMA level below which a mode/rank axis becomes a pruning candidate.",
    )
    parser.add_argument(
        "--prune-sustain-steps",
        type=int,
        default=1,
        help="Number of consecutive growth checks an axis must remain below prune_threshold before pruning.",
    )
    parser.add_argument(
        "--prune-min-steps",
        type=int,
        default=50,
        help="Minimum steps to wait after an axis's last growth event before pruning it.",
    )
    parser.add_argument("--mode-init", type=parse_mode_init, default="zero")
    parser.add_argument("--rank-init", type=parse_rank_init, default="zero")
    parser.add_argument("--generation-eval-samples", type=int, default=0)
    parser.add_argument("--generation-prompt-len", type=int, default=64)
    parser.add_argument("--generation-new-tokens", type=int, default=128)
    parser.add_argument("--generation-temperature", type=float, default=0.8)
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=20,
        help="Top-k filter for generation sampling. Use 0 to disable filtering.",
    )
    parser.add_argument(
        "--benchmark-prompt-lengths",
        type=parse_positive_int_list,
        default=(),
        help="Comma-separated prompt lengths used for streaming inference benchmarking.",
    )
    parser.add_argument("--benchmark-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    return parser


def _checkpoint_payload(
    *,
    step: int,
    model,
    optimizer,
    config: TrainingConfig,
    dataset,
    train_losses,
    val_losses,
    train_metrics,
    val_metrics,
    generation_samples,
    runtime_benchmarks,
):
    return {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "config": asdict(config),
        "vocabulary": list(dataset.tokenizer.itos),
        "source_name": dataset.source_name,
        "train_token_count": int(dataset.train_tokens.numel()),
        "val_token_count": int(dataset.val_tokens.numel()),
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "train_metrics": [
            {"step": metric_step, **asdict(metrics)}
            for metric_step, metrics in train_metrics
        ],
        "val_metrics": [
            {"step": metric_step, **asdict(metrics)}
            for metric_step, metrics in val_metrics
        ],
        "generation_samples": [asdict(sample) for sample in generation_samples],
        "runtime_benchmarks": [asdict(benchmark) for benchmark in runtime_benchmarks],
    }


def save_checkpoint(path: Path, payload) -> None:
    checkpoint_path = path.expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}", flush=True)


def write_residual_diagnostics(path: Path, diagnostics) -> None:
    diagnostics_path = path.expanduser().resolve()
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote residual diagnostics: {diagnostics_path}", flush=True)


def write_run_config(run_dir: Path, config: TrainingConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote run config: {config_path}", flush=True)


def _format_preview(text: str, *, max_chars: int = 120) -> str:
    preview = text.replace("\n", "\\n")
    if len(preview) > max_chars:
        return preview[: max_chars - 3] + "..."
    return preview


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        corpus_name=args.corpus,
        max_chars=args.max_chars,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        learning_rate=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_style=args.lr_decay_style,
        min_lr_scale=args.min_lr_scale,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        state_shape=args.state_shape,
        max_state_shape=args.max_state_shape,
        num_layers=args.num_layers,
        ffn_expansion_factor=args.ffn_expansion_factor,
        readout_type=args.readout_type,
        token_magnitude_type=args.token_magnitude_type,
        phase_type=args.phase_type,
        token_phase=args.token_phase,
        enable_self_relation=args.enable_self_relation,
        dynamic_spectral_gains=args.dynamic_spectral_gains,
        anisotropic_spectral_gains=args.anisotropic_spectral_gains,
        enable_anticipator_relation=args.enable_anticipator_relation,
        enable_cross_layer_state=args.enable_cross_layer_state,
        coupling_type=args.coupling_type,
        low_frequency_gain=args.low_frequency_gain,
        low_frequency_sigma=args.low_frequency_sigma,
        high_frequency_gain=args.high_frequency_gain,
        high_frequency_cutoff=args.high_frequency_cutoff,
        wavelet_levels=args.wavelet_levels,
        normalization_type=args.normalization_type,
        device=args.device,
        tensor_dynamic_growth=args.tensor_dynamic_growth,
        dynamic_mode_growth=args.dynamic_mode_growth,
        dynamic_rank_growth=args.dynamic_rank_growth,
        dynamic_mode_pruning=args.dynamic_mode_pruning,
        dynamic_rank_pruning=args.dynamic_rank_pruning,
        max_rank=args.max_rank,
        growth_check_interval=args.growth_check_interval,
        growth_residual_threshold=args.growth_residual_threshold,
        post_growth_cooldown_checks=args.post_growth_cooldown_checks,
        post_growth_cooldown_threshold_scale=args.post_growth_cooldown_threshold_scale,
        residual_saturate_threshold=args.residual_saturate_threshold,
        growth_residual_ema_decay=args.growth_residual_ema_decay,
        record_residual_diagnostics=args.record_residual_diagnostics,
        chunk_size=args.chunk_size,
        track_chunk_drift=args.track_chunk_drift,
        min_checks_before_first_growth=args.min_checks_before_first_growth,
        rank_growth_loss_ceiling=args.rank_growth_loss_ceiling,
        prune_threshold=args.prune_threshold,
        prune_sustain_steps=args.prune_sustain_steps,
        prune_min_steps=args.prune_min_steps,
        mode_init=args.mode_init,
        rank_init=args.rank_init,
        generation_eval_samples=args.generation_eval_samples,
        generation_prompt_len=args.generation_prompt_len,
        generation_new_tokens=args.generation_new_tokens,
        generation_temperature=args.generation_temperature,
        generation_top_k=None if args.generation_top_k <= 0 else args.generation_top_k,
        benchmark_prompt_lengths=args.benchmark_prompt_lengths,
        benchmark_new_tokens=args.benchmark_new_tokens,
        seed=args.seed,
        stateful_training=args.stateful_training,
        attention_every_k=args.attention_every_k,
        attention_num_heads=args.attention_num_heads,
        attention_window=args.attention_window,
        attention_position=args.attention_position,
    )
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative.")

    run_dir = None
    if args.run_name is not None:
        run_dir = (REPO_ROOT / args.checkpoint_dir / args.run_name).resolve()
        write_run_config(run_dir, config)

    last_state_shape = tuple(config.state_shape)

    def step_callback(
        step,
        model,
        optimizer,
        dataset,
        callback_config,
        train_losses,
        val_losses,
        train_metrics,
        val_metrics,
        device,
    ) -> None:
        nonlocal last_state_shape
        current_state_shape = tuple(callback_config.state_shape)
        if current_state_shape != last_state_shape:
            print(
                f"state_shape_event step={step} previous_state_shape={last_state_shape} state_shape={current_state_shape}",
                flush=True,
            )
            last_state_shape = current_state_shape

        latest_train = train_metrics[-1][1]
        latest_val = val_metrics[-1][1] if val_metrics and val_metrics[-1][0] == step else None
        if step == 1 or step == callback_config.steps or step % callback_config.eval_every == 0:
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")
            message = (
                f"step={step} "
                f"lr={current_lr:.6g} "
                f"train_loss={latest_train.loss:.4f} "
                f"train_acc={latest_train.accuracy:.4f} "
                f"train_ppl={latest_train.perplexity:.4f} "
                f"train_bpc={latest_train.bpc:.4f}"
            )
            if latest_val is not None:
                message += (
                    f" val_loss={latest_val.loss:.4f}"
                    f" val_acc={latest_val.accuracy:.4f}"
                    f" val_ppl={latest_val.perplexity:.4f}"
                    f" val_bpc={latest_val.bpc:.4f}"
                )
            print(message, flush=True)

        if run_dir is not None and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            payload = _checkpoint_payload(
                step=step,
                model=model,
                optimizer=optimizer,
                config=callback_config,
                dataset=dataset,
                train_losses=train_losses,
                val_losses=val_losses,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                generation_samples=[],
                runtime_benchmarks=[],
            )
            save_checkpoint(run_dir / f"checkpoint_step_{step:06d}.pt", payload)

    result = train_model(config, step_callback=step_callback)

    last_train_step, last_train_metrics = result.train_metrics[-1]
    last_eval_step, last_val_metrics = result.val_metrics[-1]
    print(
        "training finished "
        f"(device={result.device}, corpus={result.dataset.source_name}, vocab={result.dataset.vocab_size}, "
        f"train_tokens={result.dataset.train_tokens.numel()}, val_tokens={result.dataset.val_tokens.numel()}, "
        f"state_shape={result.config.state_shape})",
        flush=True,
    )
    print(
        f"final train(step={last_train_step}) loss={last_train_metrics.loss:.4f} "
        f"acc={last_train_metrics.accuracy:.4f} "
        f"ppl={last_train_metrics.perplexity:.4f} "
        f"bpc={last_train_metrics.bpc:.4f} "
        f"last eval(step={last_eval_step}) "
        f"val_loss={last_val_metrics.loss:.4f} "
        f"val_acc={last_val_metrics.accuracy:.4f} "
        f"val_ppl={last_val_metrics.perplexity:.4f} "
        f"val_bpc={last_val_metrics.bpc:.4f}",
        flush=True,
    )

    final_payload = _checkpoint_payload(
        step=config.steps,
        model=result.model,
        optimizer=None,
        config=result.config,
        dataset=result.dataset,
        train_losses=result.train_losses,
        val_losses=result.val_losses,
        train_metrics=result.train_metrics,
        val_metrics=result.val_metrics,
        generation_samples=result.generation_samples,
        runtime_benchmarks=result.runtime_benchmarks,
    )

    if run_dir is not None and (args.checkpoint_every == 0 or config.steps % args.checkpoint_every != 0):
        save_checkpoint(run_dir / f"checkpoint_step_{config.steps:06d}.pt", final_payload)

    if args.checkpoint_out is not None:
        save_checkpoint(Path(args.checkpoint_out), final_payload)

    if args.record_residual_diagnostics:
        if args.diagnostics_out is not None:
            diagnostics_path = Path(args.diagnostics_out)
        elif run_dir is not None:
            diagnostics_path = run_dir / "residual_diagnostics.json"
        else:
            diagnostics_path = REPO_ROOT / "residual_diagnostics.json"
        write_residual_diagnostics(diagnostics_path, result.residual_diagnostics)

    for benchmark in result.runtime_benchmarks:
        memory_suffix = (
            ""
            if benchmark.peak_memory_bytes is None
            else f" {benchmark.memory_metric}={benchmark.peak_memory_bytes}"
        )
        print(
            "streaming_benchmark "
            f"prompt_len={benchmark.prompt_length} "
            f"prompt_tps={benchmark.prompt_tokens_per_second:.2f} "
            f"decode_tps={benchmark.decode_tokens_per_second:.2f}"
            f"{memory_suffix}",
            flush=True,
        )

    for index, sample in enumerate(result.generation_samples, start=1):
        print(
            "generation_sample "
            f"index={index} "
            f"distinct1={sample.distinct_1:.4f} "
            f"distinct2={sample.distinct_2:.4f} "
            f"prompt='{_format_preview(sample.prompt)}' "
            f"continuation='{_format_preview(sample.continuation)}'",
            flush=True,
        )


if __name__ == "__main__":
    main()
