#!/usr/bin/env python3
"""Warm-start Lisp GRPO from a character-LM checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lisp_rl_common import (  # noqa: E402
    ResidualDiagnosticsTracker,
    load_checkpoint,
    model_from_checkpoint,
    tokenizer_from_checkpoint,
)
from reciprocator.model import ReciprocatorLM  # noqa: E402
from reciprocator.rl.curriculum import CurriculumController  # noqa: E402
from reciprocator.rl.problem_gen import default_difficulty_for_stage  # noqa: E402
from reciprocator.rl.training import LispGRPOConfig, RLStepMetrics, train_lisp_grpo  # noqa: E402
from reciprocator.training import CharTokenizer  # noqa: E402


def parse_positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--stage", type=parse_positive_int, default=1)
    parser.add_argument("--steps", type=parse_positive_int, default=100)
    parser.add_argument("--batch-size", type=parse_positive_int, default=4)
    parser.add_argument("--group-size", type=parse_positive_int, default=16)
    parser.add_argument("--max-completion-tokens", type=parse_positive_int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--stage1-wrong-reward", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-record-limit", type=int, default=8)
    parser.add_argument("--record-residual-diagnostics", action="store_true")
    parser.add_argument("--residual-diagnostic-every", type=parse_positive_int, default=1)
    parser.add_argument("--residual-diagnostic-batch-size", type=parse_positive_int, default=8)
    parser.add_argument("--residual-diagnostic-seq-len", type=parse_positive_int, default=64)
    parser.add_argument("--residual-ema-decay", type=float, default=0.8)
    parser.add_argument("--growth-residual-threshold", type=float, default=0.12)
    parser.add_argument("--residual-saturate-threshold", type=float, default=0.07)
    parser.add_argument("--prune-threshold", type=float, default=0.4)
    return parser


def grpo_config_from_args(args: argparse.Namespace, lm_config: Mapping[str, Any]) -> LispGRPOConfig:
    return LispGRPOConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_completion_tokens=args.max_completion_tokens,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        temperature=args.temperature,
        top_k=None if args.top_k <= 0 else args.top_k,
        kl_beta=args.kl_beta,
        device=args.device,
        seed=args.seed,
        hidden_size=int(lm_config["hidden_size"]),
        state_shape=tuple(lm_config["state_shape"]),
        num_layers=int(lm_config["num_layers"]),
        ffn_expansion_factor=int(lm_config.get("ffn_expansion_factor", 2)),
        readout_type=str(lm_config.get("readout_type", "phase_aware")),
        token_magnitude_type=str(lm_config.get("token_magnitude_type", "inverse_frequency_learned")),
        phase_type=str(lm_config.get("phase_type", "rope")),
        token_phase=str(lm_config.get("token_phase", "semantic")),
        enable_self_relation=bool(lm_config.get("enable_self_relation", False)),
        coupling_type=str(lm_config.get("coupling_type", "sequential")),
        normalization_type=str(lm_config.get("normalization_type", "frobenius")),
        sample_record_limit=args.sample_record_limit,
        stage1_wrong_reward=args.stage1_wrong_reward,
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_metric(path: Path, metric: RLStepMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metric), sort_keys=True) + "\n")


def write_residual_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    write_json(path, rows)


def save_checkpoint(path: Path, *, step: int, model: ReciprocatorLM, tokenizer: CharTokenizer, config: LispGRPOConfig, metrics: list[RLStepMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "vocabulary": list(tokenizer.itos),
            "step_metrics": [asdict(metric) for metric in metrics],
        },
        path,
    )
    print(f"saved rl checkpoint: {path.resolve()}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.stage <= 7:
        raise ValueError("--stage must be in [1, 7].")
    if args.checkpoint_every < 0 or args.log_every <= 0:
        raise ValueError("--checkpoint-every must be non-negative and --log-every must be positive.")

    warm_start_path = args.warm_start_checkpoint.expanduser().resolve()
    payload = load_checkpoint(warm_start_path)
    tokenizer = tokenizer_from_checkpoint(payload)
    model, lm_config = model_from_checkpoint(payload, tokenizer)
    config = grpo_config_from_args(args, lm_config)

    run_dir = (args.checkpoint_dir / args.run_name).resolve()
    metrics_path = run_dir / "metrics.jsonl"
    residual_path = run_dir / "residual_diagnostics.json"
    config_payload = {
        "warm_start_checkpoint": str(warm_start_path),
        "stage": args.stage,
        "rl_config": asdict(config),
        "lm_config": lm_config,
        "residual_diagnostics": {
            "enabled": args.record_residual_diagnostics,
            "every": args.residual_diagnostic_every,
            "batch_size": args.residual_diagnostic_batch_size,
            "seq_len": args.residual_diagnostic_seq_len,
            "ema_decay": args.residual_ema_decay,
            "growth_residual_threshold": args.growth_residual_threshold,
            "residual_saturate_threshold": args.residual_saturate_threshold,
            "prune_threshold": args.prune_threshold,
        },
    }
    write_json(run_dir / "config.json", config_payload)
    print("rl config:", flush=True)
    print(json.dumps(config_payload, indent=2, sort_keys=True), flush=True)

    stage_difficulty = default_difficulty_for_stage(args.stage)
    curriculum = CurriculumController(
        stage_difficulties={args.stage: stage_difficulty},
        current_stage=args.stage,
        harder_stage_mix=0.0,
    )
    observed_metrics: list[RLStepMetrics] = []
    residual_rows: list[dict[str, object]] = []
    residual_tracker = (
        ResidualDiagnosticsTracker(
            tokenizer=tokenizer,
            stage=args.stage,
            batch_size=args.residual_diagnostic_batch_size,
            seq_len=args.residual_diagnostic_seq_len,
            seed=args.seed + 1729,
            ema_decay=args.residual_ema_decay,
            growth_residual_threshold=args.growth_residual_threshold,
            residual_saturate_threshold=args.residual_saturate_threshold,
            prune_threshold=args.prune_threshold,
        )
        if args.record_residual_diagnostics
        else None
    )

    def step_callback(metric: RLStepMetrics, callback_model: ReciprocatorLM, _curriculum: CurriculumController) -> None:
        observed_metrics.append(metric)
        append_metric(metrics_path, metric)
        residual_row = None
        if residual_tracker is not None and (
            metric.step == 1
            or metric.step % args.residual_diagnostic_every == 0
            or metric.step == config.steps
        ):
            residual_row = residual_tracker.record(callback_model, step=metric.step)
            residual_rows.append(residual_row)
            write_residual_diagnostics(residual_path, residual_rows)
        if metric.step == 1 or metric.step % args.log_every == 0 or metric.step == config.steps:
            residual_summary = ""
            if residual_row is not None:
                residual_summary = (
                    f" residual_ema={residual_row['mode_residual_ema']} "
                    f"growth_pressure={residual_row['max_growth_pressure']:.3f} "
                    f"prune_pressure={residual_row['max_pruning_pressure']:.3f}"
                )
            print(
                f"step={metric.step} stage={metric.current_stage} "
                f"mean_reward={metric.mean_reward:.4f} success={metric.success_rate:.4f} "
                f"kl={metric.mean_kl:.5f} grad_norm={metric.grad_norm:.4f} "
                f"loss={metric.loss:.4f} errors={metric.error_counts}"
                f"{residual_summary}",
                flush=True,
            )
        if args.checkpoint_every > 0 and metric.step % args.checkpoint_every == 0:
            save_checkpoint(
                run_dir / f"checkpoint_step_{metric.step:06d}.pt",
                step=metric.step,
                model=callback_model,
                tokenizer=tokenizer,
                config=config,
                metrics=observed_metrics,
            )

    result = train_lisp_grpo(
        config,
        tokenizer=tokenizer,
        model=model,
        curriculum=curriculum,
        step_callback=step_callback,
    )
    final_metric = result.step_metrics[-1]
    save_checkpoint(
        run_dir / f"checkpoint_step_{final_metric.step:06d}.pt",
        step=final_metric.step,
        model=result.model,
        tokenizer=tokenizer,
        config=config,
        metrics=result.step_metrics,
    )
    print(
        "rl finished "
        f"(device={result.device}, stage={args.stage}, "
        f"mean_reward={final_metric.mean_reward:.4f}, success={final_metric.success_rate:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
