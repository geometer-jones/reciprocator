# Reciprocator Lab Book

## Project Context

`reciprocator` is a small Python/PyTorch research codebase for a character-level
language model built around complex-valued token lifting and a tensor-state recurrent
"Reciprocator" mixer.

- Primary language: Python
- Dependency/build system: `pyproject.toml` with `setuptools`; runtime dependency is `torch`
- Test framework: `pytest` under `tests/`
- Application entry point: [scripts/train.py](/Users/peterwei/Desktop/wokspace/reciprocator/scripts/train.py)
- Core modules:
  - [src/reciprocator/corpora.py](/Users/peterwei/Desktop/wokspace/reciprocator/src/reciprocator/corpora.py): bundled corpora and corpus loading
  - [src/reciprocator/model.py](/Users/peterwei/Desktop/wokspace/reciprocator/src/reciprocator/model.py): `TokenLift`, blocks, and `ReciprocatorLM`
  - [src/reciprocator/mixer.py](/Users/peterwei/Desktop/wokspace/reciprocator/src/reciprocator/mixer.py): sequential and spectral coupling backends plus state projection
  - [src/reciprocator/training.py](/Users/peterwei/Desktop/wokspace/reciprocator/src/reciprocator/training.py): dataset construction, training loop, evaluation, dynamic growth/pruning, generation, and benchmarking
- Data flow: bundled corpus -> tokenizer/dataset -> `TokenLift` -> Reciprocator blocks/mixer -> readout/logits -> training summaries under `runs/lab-book/`

This document is the retrospective companion to [test_plan.md](/Users/peterwei/Desktop/wokspace/reciprocator/test_plan.md). The plan is prospective. This lab book records what actually ran and what it means.

## Experimental Frame

Unless noted otherwise, the test matrix used:

- corpus: `plato_jowett`
- `max_chars=100000`
- `batch_size=8`
- `seq_len=64`
- `hidden_size=32`
- `state_shape=(2,3,4)`
- `eval_every=50`
- `eval_batches=16`
- primary metric for 500-step runs: seed-mean of final-3 `val_bpc`
- primary metric for 1000-step runs: seed-mean of final-5 `val_bpc`

Important provenance caveat: the run artifacts do not record git commit SHA or code
version. Some later reruns are therefore comparable only at the config/result level,
not at the exact code-state level.

## Current Conclusions

1. The best 500-step static configuration under the original Frobenius/sequential setup is:
   `token_magnitude_type=inverse_frequency_learned`, `phase_type=rope`, `token_phase=semantic`,
   `coupling_type=sequential`, `normalization_type=frobenius`.
   Mean final-3 `val_bpc`: `3.3033`.
2. The best 1000-step static continuation of that recipe reached mean final-5 `val_bpc = 3.1216`.
3. Sequential coupling remains the best tested backend. The spectral screen was effectively null:
   all non-sequential backends landed within `+0.004` to `+0.006` bpc of sequential and failed the
   plan's `>0.02` win rule.
4. Per-mode normalization preserved the same ranking seen under Frobenius normalization but did not
   improve the best combination. The best per-mode combo reached `3.3744`, worse than the Frobenius
   winner by `+0.0711`.
5. Dynamic mode growth produced only marginal gains. The best observed follow-up reached
   mean final-5 `val_bpc = 3.1081`, which is better than the static baseline by `0.0136`, below the
   plan's promotion threshold.
6. Rank growth is not ready. Late firing was achieved, but zero init slightly lost, mean init lost
   more, and residual init destabilized badly.
7. Pruning is also not ready. Calibrated settings never pruned anything; more aggressive settings
   pruned late and damaged performance, sometimes collapsing state shape in a way that needs separate
   debugging.

## Phase History

### Phase 0: Pilot

From [test_plan.md](/Users/peterwei/Desktop/wokspace/reciprocator/test_plan.md):

- Stable at 500 steps with no NaN/Inf
- Final `val_bpc`: `4.0774`
- Final-3 mean: `4.1453`
- This motivated the larger `100k` corpus slice, the lower `1e-3` learning rate, and later concern that the fixed optimizer protocol might be suboptimal

### Phases 1-2: Static TokenLift Search

Phase 1 established the single-axis picture against the baseline (`3.9363`):

| Variant | Mean final-3 `val_bpc` | Delta vs baseline |
|---|---:|---:|
| `inverse_frequency_learned` | 3.3640 | -0.5723 |
| `semantic` | 3.3813 | -0.5550 |
| `locked_wave` | 3.4842 | -0.4521 |
| `local_wave` | 3.4993 | -0.4370 |
| `virtual_offset` | 3.9194 | -0.0169 |
| baseline | 3.9363 | 0.0000 |

Takeaways:

- Magnitude prior clearly helps.
- Semantic token phase also clearly helps.
- Both phase variants beat baseline on their own, with `locked_wave` slightly better than `local_wave`.
- `virtual_offset` is effectively a null result.

Phase 2 then tested compositions:

| Config | Mean final-3 `val_bpc` |
|---|---:|
| `inverse_frequency_learned + semantic` (`mag_tokenphase`) | 3.3033 |
| `inverse_frequency_learned + locked_wave` (`mag_phase`) | 3.4725 |
| `inverse_frequency_learned + locked_wave + semantic` (`full_stack`) | 3.4659 |

Takeaway: the useful composition is `inverse_frequency_learned + semantic`. Adding the
phase-axis winner back in made the model worse, so the single-axis phase win did not
compose.

### Phase 2.1: Local-Wave Follow-Up

This ad hoc follow-up checked whether `local_wave` might compose better than `locked_wave`.

| Config | Mean final-3 `val_bpc` |
|---|---:|
| `inverse_frequency_learned + local_wave + semantic` | 3.4647 |
| `inverse_frequency_learned + local_wave` | 3.4733 |

Result: no. This confirmed that the best static recipe keeps `phase_type=rope`.

### Phase 2S: Spectral Coupling Screen

The sequential Phase 2 winner was re-run as the backend baseline and compared against
the new spectral couplings.

| Coupling | Mean final-3 `val_bpc` | Delta vs sequential |
|---|---:|---:|
| `sequential` | 3.3317 | 0.0000 |
| `wavelet_packet` | 3.3361 | +0.0044 |
| `wavelet_packet_max_gauge` | 3.3361 | +0.0044 |
| `fft` | 3.3368 | +0.0051 |
| `dwt` | 3.3378 | +0.0061 |

Result: no non-sequential backend beat sequential by the required `>0.02` margin.
The spectral branch should therefore be treated as closed unless the higher-dimensional
screen is finished.

### Phase 2S.3: Higher-Dimensional Spectral Check

This branch is incomplete. Only the `(8,8,8)` sequential baseline was run:

- `state_shape=(8,8,8)`, `coupling_type=sequential`
- Mean final-3 `val_bpc = 3.2767`
- `n=1` seed only

No spectral comparisons were run, so this branch is not yet evidence for or against
the dimensionality-artifact hypothesis.

### Phase 3: 1000-Step Static Baseline

The Phase 2 winner was extended to 1000 steps:

- Config: `inverse_frequency_learned + rope + semantic`, Frobenius norm, sequential coupling
- Mean final-5 `val_bpc = 3.1216`
- Mean final `val_bpc = 3.0641`

This is the main static control for all dynamic experiments.

### Phase 3.1: Residual EMA Calibration

All three EMA settings preserved the same training loss curve, as expected; the useful
output here was the residual/redundancy telemetry.

Mean per-mode residual snapshots across seeds:

| EMA alpha | Step 50 | Step 200 | Step 500 | Step 1000 |
|---|---:|---:|---:|---:|
| 0.1 | 0.2813 | 0.2347 | 0.1645 | 0.0924 |
| 0.2 | 0.2813 | 0.1969 | 0.1131 | 0.0571 |
| 0.3 | 0.2813 | 0.1670 | 0.0916 | 0.0486 |

Important structural observation:

- Residual energy is almost entirely concentrated in the first mode.
- The other two modes sit near numerical zero throughout.
- Redundancy falls into the `~0.11-0.16` range by step 1000 depending on EMA alpha.

The later dynamic runs operationally chose:

- `growth_residual_ema_decay=0.8` (`alpha=0.2`)
- `min_checks_before_first_growth=7`
- `growth_residual_threshold=0.12`
- `residual_saturate_threshold=0.07`

### Phase 3.2: Mode Growth

The original threshold sweep produced:

| Config | Mean final-5 `val_bpc` | Delta vs static | Behavior |
|---|---:|---:|---|
| `threshold=0.12`, `mode_init=zero` | 3.1104 | -0.0112 | One growth at step 350 to `(3,3,4)` |
| `threshold=0.168`, `mode_init=residual` | 3.1216 | 0.0000 | No useful improvement |
| `threshold=0.12`, `mode_init=residual` | 3.1471 | +0.0254 | Worse |
| `threshold=0.084`, `mode_init=residual` | 3.1530 | +0.0314 | Over-grew to `(4,4,5)` |

Then two one-off follow-ups were run outside the original matrix:

- [runs/lab-book/phase3_2e/summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase3_2e/summary.json): repeated the `threshold=0.12`, `mode_init=residual` case as a comparison artifact
- [runs/lab-book/phase3_2f/summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase3_2f/summary.json): tested `threshold=0.12`, `mode_init=mean`

The mean-init follow-up did best:

- Mean final-5 `val_bpc = 3.1081`
- Delta vs static: `-0.0136`
- One mode-growth event at step `350` in both seeds
- Final state shape `(3,3,4)` in both seeds

Interpretation:

- Dynamic mode growth is plausibly useful, but the current gain is small.
- The best observed recipe is: `ema_decay=0.8`, `min_checks_before_first_growth=7`,
  `growth_residual_threshold=0.12`, one mode addition around step `350`.
- `mode_init=residual` is not helping here; `mean` beat both `zero` and `residual`.

### Phase 3.3: Rank Growth

All three rank-growth runs fired at step `900`, which is already an improvement over
the earlier "fires immediately" failure mode. But the quality result was negative:

| Rank init | Mean final-5 `val_bpc` | Delta vs static |
|---|---:|---:|
| `zero` | 3.1404 | +0.0187 |
| `mean` | 3.1987 | +0.0771 |
| `residual` | 3.4995 | +0.3778 |

Interpretation:

- The late trigger is structurally better than the original failure mode.
- The mechanism still does not improve the chosen metric.
- Residual init is actively destabilizing and should not remain in the active promotion path.

### Phase 3.4: Pruning

Three calibrated pruning settings were tested, then re-run on `2026-04-22`.

Calibrated settings:

| Config | Mean final-5 `val_bpc` | Prunes observed |
|---|---:|---:|
| `prune_threshold=0.13`, sustain 2 | 3.1104 | 0 |
| `prune_threshold=0.13`, sustain 4 | 3.1104 | 0 |
| `prune_threshold=0.091`, sustain 4 | 3.1104 | 0 |

Interpretation:

- Under the thresholds that looked sensible from Phase 3.1, pruning never happened.
- These runs therefore just reproduced the mode-growth baseline.

More aggressive follow-ups in [runs/lab-book/phase3_4_hi/summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase3_4_hi/summary.json) did prune, but the result was clearly bad:

- `prune_threshold=0.16`, sustain 2: one seed pruned at step `1000` and collapsed to `[3,3]`; mean final-5 `3.2098`
- `prune_threshold=0.18`, sustain 4: pruned to `[3,3]` late; mean final-5 `3.3239`
- `prune_threshold=0.20`, sustain 4: pruned to `[3,3]` or `[3,4]`; mean final-5 `3.3243`

The shape transitions `[3,3,4] -> [3,3]` and `[3,3,4] -> [3,4]` are not obviously
aligned with the phase intent and should be treated as a debugging target before any
more pruning sweeps are run.

### Phases 5-6: Per-Mode Normalization

Phase 5 repeated the Phase 1 axis scan under `normalization_type=per_mode`.
The ranking stayed the same:

| Variant | Mean final-3 `val_bpc` |
|---|---:|
| `inverse_frequency_learned` | 3.4059 |
| `semantic` | 3.4375 |
| `locked_wave` | 3.4987 |
| `local_wave` | 3.5172 |
| baseline | 4.0249 |
| `virtual_offset` | 4.0318 |

Phase 6 repeated the composition step:

| Config | Mean final-3 `val_bpc` |
|---|---:|
| `inverse_frequency_learned + semantic` | 3.3744 |
| `inverse_frequency_learned + locked_wave` | 3.4907 |
| `inverse_frequency_learned + locked_wave + semantic` | 3.5451 |

Interpretation:

- Per-mode normalization preserved the overall ordering.
- It did not beat the Frobenius winner.
- Frobenius should remain the default unless a later scaling experiment changes the picture.

## Best-Known Recipes

### Best 500-step static recipe

- `token_magnitude_type=inverse_frequency_learned`
- `phase_type=rope`
- `token_phase=semantic`
- `coupling_type=sequential`
- `normalization_type=frobenius`
- mean final-3 `val_bpc = 3.3033`

### Best 1000-step static recipe

Same config as above.

- mean final-5 `val_bpc = 3.1216`
- mean final `val_bpc = 3.0641`

### Best dynamic recipe tested so far

Same base config plus:

- `dynamic_mode_growth=True`
- `growth_residual_ema_decay=0.8`
- `min_checks_before_first_growth=7`
- `growth_residual_threshold=0.12`
- `mode_init=mean`
- one growth at step `350`

Outcome:

- mean final-5 `val_bpc = 3.1081`
- improvement over static: `0.0136`
- below the plan's `0.02` promotion threshold

## Closed, Pending, and Incomplete Branches

Closed by result:

- Spectral backend promotion under the original state shape
- `virtual_offset` token phase
- Rank-growth promotion in its current form
- Per-mode normalization as a default replacement for Frobenius

Incomplete:

- Phase 2S.3 higher-dimensional spectral screen
- Phase 3.5 full-system dynamic run
- Phase 4 / 4.1 / 4.2 ablations
- Phase 7 optimizer validation
- Phases 8-10 scaling and benchmarking branches
- Post-hoc generation and streaming benchmarks as promotion gates

## Recommended Revisions To `test_plan.md`

1. Split the plan from the record. Keep `test_plan.md` strictly prospective and treat this file as the retrospective lab notebook. Right now the plan contains completed results, pending branches, and ad hoc reruns in one place.
2. Add provenance to every run artifact: git commit SHA, run date, config digest, and corpus hash. Without this, comparisons across reruns are weaker than they should be.
3. Move optimizer validation earlier. The plan itself says the fixed `1e-3` protocol is legacy; Phase 7 should happen before more large sweeps.
4. Tighten the promotion rule. A hard `>0.02` bpc cutoff is close to the scale of some rerun differences; require both a margin and a noise check.
5. Mark the spectral branch as closed unless Phase 2S.3 is fully run. If the higher-dimensional check stays incomplete, remove it from the active critical path.
6. Record the current dynamic-mode recipe explicitly in the plan and downgrade it from "candidate winner" to "small-effect exploratory result." The best observed gain is real but below the promotion threshold.
7. Remove rank growth from the active comparison matrix until the trigger/init path is redesigned. The late trigger is better, but the quality result is still negative.
8. Reframe pruning as a debugging task, not a sweep task. Add invariants: pruning must remove exactly one intended axis, preserve rank semantics, and log which axis was pruned. The observed shape collapses need explanation first.
9. Update the run summary table so it matches reality. It should include `phase2_1`, `phase3_2e`, `phase3_2f`, `phase3_4_hi`, and `phase3_4_rerun_20260422`, plus real status markers for completed and closed branches.
10. Make the generation and streaming gates real or mark them pending. The plan says promotion requires them, but the current result summaries do not include those artifacts.
11. Mark blocked prerequisites inline. Phase 4.1 still depends on plumbing `phase_scale` through the CLI/training surface.
12. Reorder the Phase 4 subsections. `4.2` currently appears before `4.1`, which makes the plan harder to scan.
