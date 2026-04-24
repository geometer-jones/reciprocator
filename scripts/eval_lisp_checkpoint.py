#!/usr/bin/env python3
"""Evaluate Lisp checkpoint exactness on held-out generated problems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lisp_rl_common import (  # noqa: E402
    ResidualDiagnosticsTracker,
    load_checkpoint,
    model_from_checkpoint,
    resolve_device,
    tokenizer_from_checkpoint,
)
from reciprocator.rl.problem_gen import ProblemExample, ProblemGenerator, default_difficulty_for_stage  # noqa: E402
from reciprocator.rl.reward import RewardFunction  # noqa: E402
from reciprocator.rl.training import _sample_completion  # noqa: E402


def parse_positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--stage", type=parse_positive_int, default=1)
    parser.add_argument("--num-problems", type=parse_positive_int, default=100)
    parser.add_argument("--pass-k", type=int, default=16)
    parser.add_argument("--max-completion-tokens", type=parse_positive_int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--stage1-wrong-reward", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record-residual-diagnostics", action="store_true")
    parser.add_argument("--residual-diagnostic-batch-size", type=parse_positive_int, default=8)
    parser.add_argument("--residual-diagnostic-seq-len", type=parse_positive_int, default=64)
    parser.add_argument("--residual-ema-decay", type=float, default=0.8)
    parser.add_argument("--growth-residual-threshold", type=float, default=0.12)
    parser.add_argument("--residual-saturate-threshold", type=float, default=0.07)
    parser.add_argument("--prune-threshold", type=float, default=0.4)
    parser.add_argument("--sample-record-limit", type=int, default=8)
    return parser


def generate_problems(*, stage: int, count: int, seed: int) -> list[ProblemExample]:
    rng = random.Random(seed)
    generator = ProblemGenerator(rng)
    difficulty = default_difficulty_for_stage(stage)
    return [generator.generate_problem(difficulty) for _ in range(count)]


def score_completion(
    model,
    tokenizer,
    problem: ProblemExample,
    reward_function: RewardFunction,
    *,
    max_completion_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device,
) -> tuple[str, bool, str, float]:
    completion, _ = _sample_completion(
        model,
        tokenizer,
        problem.prompt_expression + "\n",
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
    reward = reward_function.score_output(problem, completion)
    return completion, reward.correct, reward.error_type, reward.reward


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path, problems: list[ProblemExample]) -> dict[str, object]:
    payload = load_checkpoint(checkpoint)
    tokenizer = tokenizer_from_checkpoint(payload)
    model, lm_config = model_from_checkpoint(payload, tokenizer)
    device = resolve_device(args.device)
    model.to(device)
    model.eval()
    reward_function = RewardFunction(stage_one_wrong_reward=args.stage1_wrong_reward)
    top_k = None if args.top_k <= 0 else args.top_k

    greedy_correct = 0
    pass_correct = 0
    greedy_error_counts: dict[str, int] = {}
    pass_error_counts: dict[str, int] = {}
    samples = []

    for index, problem in enumerate(problems):
        greedy_completion, is_correct, error_type, reward = score_completion(
            model,
            tokenizer,
            problem,
            reward_function,
            max_completion_tokens=args.max_completion_tokens,
            temperature=0.0,
            top_k=None,
            device=device,
        )
        greedy_correct += int(is_correct)
        greedy_error_counts[error_type] = greedy_error_counts.get(error_type, 0) + 1

        pass_outputs = []
        found = is_correct
        for _ in range(max(0, args.pass_k)):
            completion, sample_correct, sample_error, sample_reward = score_completion(
                model,
                tokenizer,
                problem,
                reward_function,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                top_k=top_k,
                device=device,
            )
            found = found or sample_correct
            pass_error_counts[sample_error] = pass_error_counts.get(sample_error, 0) + 1
            if len(pass_outputs) < 4:
                pass_outputs.append(
                    {
                        "completion": completion,
                        "correct": sample_correct,
                        "error_type": sample_error,
                        "reward": sample_reward,
                    }
                )
        pass_correct += int(found)

        if len(samples) < args.sample_record_limit:
            samples.append(
                {
                    "index": index,
                    "prompt_expression": problem.prompt_expression,
                    "expected_result": problem.expected_result_text,
                    "greedy_completion": greedy_completion,
                    "greedy_correct": is_correct,
                    "greedy_error_type": error_type,
                    "greedy_reward": reward,
                    "sampled_completions": pass_outputs,
                }
            )

    residual_row = None
    if args.record_residual_diagnostics:
        tracker = ResidualDiagnosticsTracker(
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
        residual_row = tracker.record(model, step=int(payload.get("step", 0) or 0))

    total = len(problems)
    return {
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "checkpoint_step": payload.get("step"),
        "stage": args.stage,
        "num_problems": total,
        "pass_k": args.pass_k,
        "greedy_exact": greedy_correct / total,
        "pass_at_k": pass_correct / total,
        "greedy_correct": greedy_correct,
        "pass_correct": pass_correct,
        "greedy_error_counts": greedy_error_counts,
        "sample_error_counts": pass_error_counts,
        "residual_diagnostics": residual_row,
        "lm_config": {
            "hidden_size": lm_config.get("hidden_size"),
            "state_shape": lm_config.get("state_shape"),
            "num_layers": lm_config.get("num_layers"),
            "normalization_type": lm_config.get("normalization_type"),
            "token_magnitude_type": lm_config.get("token_magnitude_type"),
            "phase_type": lm_config.get("phase_type"),
            "readout_type": lm_config.get("readout_type"),
        },
        "samples": samples,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.stage <= 7:
        raise ValueError("--stage must be in [1, 7].")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative.")
    if args.pass_k < 0:
        raise ValueError("--pass-k must be non-negative.")

    problems = generate_problems(stage=args.stage, count=args.num_problems, seed=args.seed)
    results = [evaluate_checkpoint(args, checkpoint, problems) for checkpoint in args.checkpoint]
    for result in results:
        residual = result["residual_diagnostics"] or {}
        residual_summary = ""
        if residual:
            residual_summary = (
                f" residual_ema={residual['mode_residual_ema']} "
                f"growth_pressure={residual['max_growth_pressure']:.3f} "
                f"prune_pressure={residual['max_pruning_pressure']:.3f}"
            )
        print(
            f"checkpoint={Path(result['checkpoint']).name} step={result['checkpoint_step']} "
            f"stage={result['stage']} n={result['num_problems']} "
            f"greedy_exact={result['greedy_exact']:.4f} "
            f"pass@{result['pass_k']}={result['pass_at_k']:.4f}"
            f"{residual_summary}",
            flush=True,
        )

    payload = {
        "config": {
            "stage": args.stage,
            "num_problems": args.num_problems,
            "pass_k": args.pass_k,
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "stage1_wrong_reward": args.stage1_wrong_reward,
            "prune_threshold": args.prune_threshold,
            "seed": args.seed,
        },
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
