# RL Mathematical Reasoning — Lab Book

This document is the retrospective companion to [`test-plan.md`](./test-plan.md).
The plan is prospective. This lab book records what actually ran and what it means.

## Experimental Frame

Unless noted otherwise:

- model: `ReciprocatorLM` with `inverse_frequency_learned` / `rope` / `semantic`,
  Frobenius norm, sequential coupling
- tokenizer: character-level, built from the Lisp dialect character set
- RL method: GRPO with `grpo_group_size=4`, `kl_coeff=0.01`
- primary metric: accuracy by curriculum stage
- reporting: mean ± std across seeds

## Phase History

### R0 - Evaluator and Problem Generator Validation

Run timestamp: 2026-04-23 19:45 PDT.

Result: passed.

Validation:

- Added and ran `tests/test_rl_phase_r0.py`.
- Evaluator forms, malformed-input errors, eval errors, and stage gates passed.
- Problem generator produced syntactically valid problems and matched a fresh evaluator
  for 1000 examples per stage across stages 1-7.
- Reward contract passed for evaluable stages; stage 6 proof rewards passed the
  proof-specific complete/invalid/garbage checks.
- Full focused command: `python3 -m pytest tests/test_rl_phase_r0.py`.

Fixes made during R0:

- `symbolic_equivalence()` now treats unsupported sampled symbolic evaluations as
  non-equivalent instead of leaking `EvalError`.
- Stage 3 quote rewards now return the intended `0.2` eval-error score for parseable
  outputs that are not raw quoted data and cannot be evaluated.

### R1 - GRPO Smoke Test Diagnostic

Run timestamp: 2026-04-23 PDT.

Result: diagnostic passed.

Artifact root: `runs/rl/phase_r1_20260423_diagnostic/`.

Configuration:

- 3 seeds: 0, 1, 2.
- 200 steps per seed.
- `hidden_size=32`, `state_shape=(2, 3, 4)`, `group_size=4`.
- Stage 1 arithmetic only, using generator `depth=2` because this implementation's
  `depth=1` emits numeric literals rather than one-operator arithmetic problems.
- Local-runtime deviations: `batch_size=2` instead of 32, `max_completion_tokens=4`
  instead of 32, and phase-trajectory recording disabled. R1 does not evaluate phase
  statistics; it checks gradients, reward movement, parseability, and parameter
  finiteness.

Decision checks:

- Gradient norms were finite and non-zero at every step for all seeds.
- Mean reward changed by more than 0.1 from first to final step for all seeds:
  seed 0 `0.1500 -> 0.3875`, seed 1 `0.0500 -> 0.2000`, seed 2 `0.1250 -> 0.2375`.
- Final-step outputs included parseable outputs for all seeds (`8/8` parseable in
  the final sampled batch for each seed).
- Final model parameters were finite for all seeds.

Notes:

- A 10-step moving-window reward trend was noisier: seed 1 changed by `0.0900` and
  seed 2 by `0.0800`. The literal first-to-final R1 rule passed, but the trend is
  still weak enough that R2 should retain close reward-curve monitoring.
- The RL loop now records `grad_norm` directly in `RLStepMetrics`, and completion
  sequence statistics are computed with one teacher-forced forward pass per sample
  instead of replaying one token at a time.

## Current Conclusions

R0 passed. The evaluator, generator, and reward infrastructure are sufficient for
RL training. R1 diagnostic passed the literal smoke-test checks, so the gradient
path works and reward is moving, but the weak 10-step trend on two seeds argues for
close reward-curve monitoring in R2 before increasing task difficulty.
