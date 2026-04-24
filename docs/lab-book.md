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
- Data flow: bundled corpus -> tokenizer/dataset -> `TokenLift` -> Reciprocator blocks/mixer -> readout/logits -> training summaries under `runs/`

This document is the retrospective companion to [docs/test-plan.md](/Users/peterwei/Desktop/wokspace/reciprocator/docs/test-plan.md). The plan is prospective. This lab book records what actually ran and what it means.

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
2. Phase 4 promoted `readout_type=phase_aware`, and Phase 4C showed it composes with
   `num_layers=2`. The current best static quality recipe is
   `phase_aware + num_layers=2`, with mean final-3 `val_bpc = 3.2214` at 500 steps and
   mean final-5 `val_bpc = 3.0795` at 1000 steps.
3. The best 1000-step static continuation of the pre-Phase 4 recipe reached mean final-5 `val_bpc = 3.1216`.
4. Sequential coupling remains the default backend. The original spectral screen was null, the
   higher-dimensional `(8,8,8)` check did not overturn it, and the Phase 2D dynamic backend screen
   also lost: `dwt` was `+0.0380` bpc and `fft` was `+0.0542` bpc worse than sequential under the
   dynamic rank/mode recipe.
5. Per-mode normalization preserved the same ranking seen under Frobenius normalization but did not
   improve the best combination. The best per-mode combo reached `3.3744`, worse than the Frobenius
   winner by `+0.0711`.
6. Dynamic mode growth produced only marginal single-layer gains, and did not transfer
   to the two-layer phase-aware recipe. Phase 4.2 mode growth lost to the matched
   two-layer static control by `+0.0113` bpc and was slower.
7. Rank growth is not ready as a default. Late firing was achieved, and Phase 2D made rank growth
   fire in every seed, but the dynamic rank/mode sequential result (`3.1234`) did not improve the
   1000-step static control (`3.1216`) and remained worse than the best mode-only dynamic result
   (`3.1081`).
8. Pruning is also not ready. Calibrated settings never pruned anything; more aggressive settings
   pruned late and damaged performance, sometimes collapsing state shape in a way that needs separate
   debugging.
9. Phase 7 rejected the proposed warmup/cosine/clip/decay optimizer protocol for this 500-step
   baseline. The legacy fixed-LR protocol won by `0.2924` to `0.3723` bpc, so new short-budget
   phases should keep `lr=1e-3`, constant LR, no clipping, and no weight decay unless a later
   longer-budget phase revalidates scheduling.

## Phase History

### Phase 0: Pilot

From [docs/test-plan.md](/Users/peterwei/Desktop/wokspace/reciprocator/docs/test-plan.md):

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
This made the higher-dimensional check the only remaining reason to keep the spectral
branch open.

### Phase 2S.2: FFT Filter Calibration

This conditional branch was run post-hoc on `2026-04-23` for `coupling_type=fft` only.
It used the historical Phase 2S stack rather than the newer Phase 4 defaults:
`readout_type=magnitude`, `inverse_frequency_learned + rope + semantic`, Frobenius
norm, `state_shape=(2,3,4)`, and no growth.

| Run | Filter preset | Mean final-3 `val_bpc` | Delta vs 2S sequential | Delta vs 2S FFT default | Mean seconds/run | Mean train tokens/sec |
|---|---|---:|---:|---:|---:|---:|
| 2S.2a | gentle smoothing | 3.3380 | +0.0063 | +0.0012 | 38.2 | 6717.4 |
| 2S.2b | low-frequency emphasis | 3.3379 | +0.0062 | +0.0011 | 40.4 | 6350.4 |
| 2S.2c | aggressive low-pass | 3.3378 | +0.0061 | +0.0010 | 41.0 | 6248.8 |

Decision: none of the FFT filter presets improved on the inherited FFT default, and
all remained worse than the Phase 2S sequential baseline (`3.3317`). This keeps the
spectral branch closed.

Artifacts:

- [summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase2_s2_fft_20260423/summary.json)
- [aggregate_summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase2_s2_fft_20260423/aggregate_summary.json)

### Phase 2S.3: Higher-Dimensional Spectral Check

This branch reran the Phase 2S backend screen at `state_shape=(8,8,8)` to test whether
the original spectral null was only caused by tiny mode sizes.

| Coupling | Mean final-3 `val_bpc` | Delta vs sequential |
|---|---:|---:|
| `sequential` | 3.2882 | 0.0000 |
| `wavelet_packet` | 3.2726 | -0.0156 |
| `wavelet_packet_max_gauge` | 3.2726 | -0.0156 |
| `dwt` | 3.2738 | -0.0144 |
| `fft` | 3.2740 | -0.0142 |

Result: spectral backends did improve at the larger state shape, but not enough to
clear the plan's `>0.02` rule. This closes the dimensionality-artifact hypothesis for
now: larger state modes helped spectral coupling slightly, but did not make it the
default.

Implementation note: `wavelet_packet` and `wavelet_packet_max_gauge` are almost the
same path in [src/reciprocator/mixer.py](/Users/peterwei/Desktop/wokspace/reciprocator/src/reciprocator/mixer.py). The max-gauge variant adds a phase-coherence term to the
best-basis cost; it does not use a different packet transform. At `(8,8,8)`, both
variants landed on the same mean metric.

### Phase 2D: Dynamic Rank/Mode Backend Adaptation

This phase tested the expanded dynamic recipe at `state_shape=(4,8,8,8)` across
`sequential`, `fft`, and `dwt`. Both wavelet-packet variants were excluded.

Dynamic settings:

- `dynamic_mode_growth=True`
- `dynamic_rank_growth=True`
- `dynamic_mode_pruning=True`
- `dynamic_rank_pruning=True`
- `mode_init=mean`
- `rank_init=zero`
- `max_state_shape=(6,10,10,10)`
- `max_rank=5`

| Coupling | Mean final-5 `val_bpc` | Delta vs sequential | Mode growth seeds | Rank growth seeds | Prune seeds |
|---|---:|---:|---:|---:|---:|
| `sequential` | 3.1234 | 0.0000 | 3/3 | 3/3 | 0/3 |
| `dwt` | 3.1614 | +0.0380 | 3/3 | 3/3 | 0/3 |
| `fft` | 3.1776 | +0.0542 | 3/3 | 3/3 | 0/3 |

Event pattern: every seed grew one mode at step `350`, then appended a rank axis at
step `400`, ending at `state_shape=(5,8,8,8,2)`. Pruning was enabled but did not fire
under the conservative thresholds.

Decision: no spectral dynamic backend is promoted. The sequential dynamic rank/mode
control also does not become the best dynamic recipe: it is effectively tied with, and
slightly worse than, the 1000-step static baseline (`3.1216`) and worse than the best
mode-only dynamic run (`3.1081`). Rank growth now fires reliably, but the current
zero-initialized rank addition does not buy quality.

Artifacts:

- [summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase2d_dynamic_backend_20260423/summary.json)
- [aggregate_summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase2d_dynamic_backend_20260423/aggregate_summary.json)

### Phase 2D.X: Expanded Spectral-Only Rank-10 Rerun

This follow-up reran only the spectral backends, `fft` and `dwt`, with more state
capacity and a higher rank-growth ceiling:

- `state_shape=(5,10,10,10,2)`
- `max_state_shape=(6,12,12,12,3)`
- `max_rank=10`
- same Phase 2D dynamic settings otherwise

| Coupling | Mean final-5 `val_bpc` | Delta vs FFT | Mean seconds/run | Mean train tokens/sec | Mode growth seeds | Rank growth seeds |
|---|---:|---:|---:|---:|---:|---:|
| `dwt` | 3.1664 | -0.0002 | 1135.6 | 460.9 | 3/3 | 0/3 |
| `fft` | 3.1666 | 0.0000 | 1259.9 | 407.2 | 3/3 | 0/3 |

Event pattern: every seed grew the last existing mode at step `350`, from
`(5,10,10,10,2)` to `(5,10,10,10,3)`. No seed appended a new rank axis despite
`max_rank=10`; the cap was higher, but the rank-growth trigger did not fire.

Decision: expanded spectral capacity did not change the conclusion. DWT and FFT tied
on quality, DWT was about 13% faster on this CPU run, and neither beat the Phase 2D
sequential dynamic control (`3.1234`) or the best mode-only dynamic recipe (`3.1081`).

Artifacts:

- [summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase2d_spectral_expanded_rank10_20260423/summary.json)
- [aggregate_summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase2d_spectral_expanded_rank10_20260423/aggregate_summary.json)

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

- [runs/phase3_2e/summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/phase3_2e/summary.json): repeated the `threshold=0.12`, `mode_init=residual` case as a comparison artifact
- [runs/phase3_2f/summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/phase3_2f/summary.json): tested `threshold=0.12`, `mode_init=mean`

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

More aggressive follow-ups in [runs/phase3_4_hi/summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/phase3_4_hi/summary.json) did prune, but the result was clearly bad:

- `prune_threshold=0.16`, sustain 2: one seed pruned at step `1000` and collapsed to `[3,3]`; mean final-5 `3.2098`
- `prune_threshold=0.18`, sustain 4: pruned to `[3,3]` late; mean final-5 `3.3239`
- `prune_threshold=0.20`, sustain 4: pruned to `[3,3]` or `[3,4]`; mean final-5 `3.3243`

The shape transitions `[3,3,4] -> [3,3]` and `[3,3,4] -> [3,4]` are not obviously
aligned with the phase intent and should be treated as a debugging target before any
more pruning sweeps are run.

### Phase 4: Static Ablations

Phase 4 was run on `2026-04-23` as a two-seed static ablation matrix using the Phase 2
sequential winner (`inverse_frequency_learned + rope + semantic`, Frobenius norm,
sequential coupling). I added a same-phase control run because the current parser uses
checkpoint `val_metrics` directly and gives cleaner same-seed deltas than the older
Phase 2 summaries.

| Run | Variant | Mean final-3 `val_bpc` | Delta vs control | Mean seconds/run | Mean train tokens/sec |
|---|---|---:|---:|---:|---:|
| 4.0 | control | 3.3553 | 0.0000 | 66.6 | 3844.3 |
| 14 | `token_magnitude_type=inverse_frequency` | 3.3552 | -0.0001 | 69.0 | 3718.7 |
| 15 | `readout_type=phase_aware` | 3.2905 | -0.0648 | 71.3 | 3588.9 |
| 16 | `num_layers=2` | 3.2947 | -0.0606 | 140.2 | 1826.1 |
| 17 | `token_phase=semantic_virtual_offset` | 3.3485 | -0.0068 | 75.1 | 3410.6 |
| 18 | `enable_self_relation=True` | 3.3543 | -0.0010 | 78.6 | 3255.9 |
| 18b | `enable_anticipator_relation=True` | 3.3540 | -0.0013 | 78.1 | 3278.2 |

Decision:

- Promote `readout_type=phase_aware` as the best single static ablation. It clears the
  0.02 bpc threshold and keeps most of the control throughput.
- `num_layers=2` also clears the quality threshold, but it is slightly worse than
  phase-aware readout and roughly halves throughput. It is a real depth signal, not the
  current best single change.
- `inverse_frequency` without learned residual, `semantic_virtual_offset`,
  `self_relation`, and `anticipator_relation` are null results at this budget.
- Before Phase 4.2, run a small combination check for `phase_aware + num_layers=2`.
  If that combination wins, it is the better static control for any multi-layer dynamic
  sanity check.

Artifacts:

- [summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase4_static_ablations_20260423/summary.json)
- [aggregate_summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase4_static_ablations_20260423/aggregate_summary.json)

### Phase 4C: Phase-Aware Readout + Depth Combination

This two-seed follow-up tested whether the two Phase 4 wins compose:
`readout_type=phase_aware` and `num_layers=2`, with all other Phase 4 settings fixed.

| Config | Mean final-3 `val_bpc` | Delta vs Phase 4 control | Delta vs phase-aware only | Delta vs depth-only | Mean seconds/run | Mean train tokens/sec |
|---|---:|---:|---:|---:|---:|---:|
| `phase_aware + num_layers=2` | 3.2214 | -0.1339 | -0.0691 | -0.0734 | 137.4 | 1863.2 |

Per-seed final-3 `val_bpc`: `3.2012`, `3.2416`.

Decision: the two improvements compose strongly. `phase_aware + num_layers=2` becomes
the new best static control for quality-oriented experiments. The throughput cost is
real, about half the single-layer control, so keep single-layer `phase_aware` as the
efficient/default-small recipe when speed matters.

Implication: Phase 4.2 is now warranted. The relevant question is no longer whether
static depth helps; it does. The next dynamic sanity check should compare a 2-layer
static control against the best single-layer dynamic-growth recipe, while logging
per-layer residual traces before interpreting any growth result.

Artifacts:

- [summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase4_combo_phase_aware_depth2_20260423/summary.json)
- [aggregate_summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase4_combo_phase_aware_depth2_20260423/aggregate_summary.json)

### Phase 4.2: Multi-Layer Mode-Growth Sanity Check

Phase 4.2 ran the two-layer phase-aware quality recipe for 1000 steps, with and without
the best single-layer mode-growth recipe from Phase 3.2f. Rank growth and pruning were
left disabled.

Mode-growth settings for 4.2b:

- `dynamic_mode_growth=True`
- `growth_residual_threshold=0.12`
- `growth_residual_ema_decay=0.8`
- `min_checks_before_first_growth=7`
- `mode_init=mean`
- `max_state_shape=(6,6,6)`

| Run | Config | Mean final-5 `val_bpc` | Delta vs static | Mean final `val_bpc` | Mean seconds/run | Mean train tokens/sec | Growth events |
|---|---|---:|---:|---:|---:|---:|---|
| 4.2a | `phase_aware + num_layers=2`, static | 3.0795 | 0.0000 | 3.0225 | 307.7 | 1667.9 | none |
| 4.2b | 4.2a + mode growth | 3.0909 | +0.0113 | 3.0594 | 367.7 | 1394.7 | step `350` in both seeds |

Per-layer residual telemetry:

- Before growth, layer 2 had the larger mode-0 residual in both dynamic seeds. At step
  `350`, seed 0 layer EMAs were roughly `[0.0825, 0, 0]` and `[0.1591, 0, 0]`; seed 1
  was `[0.1399, 0, 0]` and `[0.2057, 0, 0]`.
- The averaged trigger therefore hid a real layer difference, but not a contradictory
  one: both layers pointed at the same mode.
- After growth, residuals redistributed into the newly added mode and converged without
  grow/prune churn. Final shapes were `(3,3,4)` for both dynamic seeds.

Decision: do not promote mode growth for multi-layer phase-aware models. The event
timing was structurally sane, but quality was worse than the matched static control and
runtime was lower. Keep dynamic-growth claims scoped to the single-layer exploratory
branch until a new trigger or init rule demonstrates a clear multi-layer gain.

Artifacts:

- [summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase4_2_multilayer_growth_20260423/summary.json)
- [aggregate_summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase4_2_multilayer_growth_20260423/aggregate_summary.json)

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

### Phase 7: Optimization Protocol Validation

Phase 7 tested whether the fixed optimizer used in Phases 1-6 should be replaced by
the proposed warmup + cosine schedule, with optional gradient clipping and weight
decay. This phase used the Phase 1 baseline config: `token_magnitude_type=learned`,
`phase_type=rope`, `token_phase=none`, Frobenius norm, sequential coupling.

| Config | Protocol | Mean final-3 `val_bpc` | Delta vs 7.1 |
|---|---|---:|---:|
| 7.1 | legacy fixed LR | 4.0755 | 0.0000 |
| 7.2 | warmup + cosine | 4.4162 | +0.3407 |
| 7.3 | warmup + cosine + clipping | 4.3679 | +0.2924 |
| 7.4 | warmup + cosine + weight decay | 4.4479 | +0.3723 |
| 7.5 | warmup + cosine + clipping + weight decay | 4.4461 | +0.3706 |

Stability notes:

- No run showed NaN/Inf in the logs.
- `7.5` was stable and seed-consistent, but much worse on loss.
- The scheduled variants decayed the learning rate within a 500-step budget and
  under-trained this small baseline.

Decision: keep the legacy optimizer protocol for new short-budget runs:

- `learning_rate=1e-3`
- `lr_warmup_steps=0`
- `lr_decay_style=constant`
- `grad_clip_norm=None`
- `weight_decay=0.0`

The proposed full protocol should not become the default for the current 500-step
matrix. A longer-budget phase can revisit scheduling, but Phase 7 is decisive for the
existing short-run test plan.

### Phase 8: State-Shape Exploration (`greek_classics`)

Phase 8 reran the Phase 4C static recipe on `greek_classics` rather than the older
`plato_jowett` setup, because the current active plan now uses `greek_classics` as the
bundled corpus. Treat the results as within-corpus comparisons only; the absolute bpc
values are not directly interchangeable with the earlier `plato_jowett` branches.

The base recipe was:

- `readout_type=phase_aware`
- `num_layers=2`
- `token_magnitude_type=inverse_frequency_learned`
- `phase_type=rope`
- `token_phase=semantic`
- `normalization_type=frobenius`

This phase compared five non-duplicate state shapes across both `sequential` and `fft`
coupling, with two seeds per condition.

| Shape | Coupling | Mean final-3 `val_bpc` | Delta vs 8.A1 sequential | Delta vs same-shape sequential | Mean seconds/run | Mean train tokens/sec |
|---|---|---:|---:|---:|---:|---:|
| `(2,3,4)` | sequential | 3.0815 | 0.0000 | 0.0000 | 176.4 | 1451.4 |
| `(2,3,4)` | fft | 3.0785 | -0.0030 | -0.0030 | 92.4 | 2770.1 |
| `(4,3,2)` | sequential | 3.0794 | -0.0021 | 0.0000 | 165.7 | 1545.1 |
| `(4,3,2)` | fft | 3.0766 | -0.0049 | -0.0028 | 86.0 | 2977.7 |
| `(2,4,3)` | sequential | 3.0804 | -0.0011 | 0.0000 | 153.1 | 1672.7 |
| `(2,4,3)` | fft | 3.0786 | -0.0029 | -0.0017 | 85.2 | 3005.4 |
| `(4,6)` | sequential | 3.0740 | -0.0075 | 0.0000 | 123.2 | 2077.7 |
| `(4,6)` | fft | 3.0784 | -0.0031 | +0.0044 | 80.4 | 3184.1 |
| `(24,)` | sequential | 3.0993 | +0.0178 | 0.0000 | 111.9 | 2288.1 |
| `(24,)` | fft | 3.0760 | -0.0055 | -0.0233 | 71.5 | 3582.1 |

Interpretation:

- Mode ordering was effectively a null result at this budget. The three rank-3
  permutations stayed within `0.0021` bpc of each other under `sequential`.
- The best overall condition was the rank-2 same-capacity shape `(4,6)` under
  `sequential`, with mean final-3 `val_bpc = 3.0740`.
- That win is real but small: `-0.0075` bpc relative to the rank-3 control `(2,3,4)`,
  below the plan's `0.02` promotion threshold.
- The rank-1 vector floor `(24,)` was bad under `sequential` (`+0.0178` bpc vs control),
  so the tensorized state still appears to matter on the direct recurrent path.
- The best FFT result came from that same rank-1 vector floor, and it beat the
  same-shape sequential run by `-0.0233` bpc. But it still did not beat the best
  overall sequential result `(4,6)`, so FFT does not become the default backend.

Decision:

- Do not promote a new default state shape from Phase 8 alone. `(4,6)` is the most
  promising follow-up shape on `greek_classics`, but its margin is below threshold.
- Do not reopen the FFT branch. FFT showed a narrow advantage only on the degenerate
  rank-1 vector state, not as the overall winner.
- If later scaling work continues on `greek_classics`, `(4,6)` is the most justified
  alternative to retest first.

Artifacts:

- [summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase8_state_shape_seq_fft_greek_classics_20260423/summary.json)
- [aggregate_summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase8_state_shape_seq_fft_greek_classics_20260423/aggregate_summary.json)

### Phase 9: Sequence-Length Scaling (`greek_classics`)

Phase 9 kept the promoted Phase 4C-style static control on `greek_classics` and
varied only `seq_len`. The recipe was:

- `state_shape=(2,3,4)`
- `readout_type=phase_aware`
- `num_layers=2`
- `token_magnitude_type=inverse_frequency_learned`
- `phase_type=rope`
- `token_phase=semantic`
- `normalization_type=frobenius`
- `coupling_type=sequential`
- `learning_rate=1e-3`
- `lr_warmup_steps=0`
- `lr_decay_style=constant`
- `grad_clip_norm=None`
- `weight_decay=0.0`

This was run as a fixed-step comparison: `batch_size=8`, `steps=500`, two seeds per
condition. That means longer `seq_len` also saw more total training tokens, so this
phase measures the plan's practical fixed-step operating point rather than isolating
"longer context helps" as a clean causal claim.

| Seq len | Eval batches | Train tokens/run | Mean final-3 `val_bpc` | Delta vs `64` | Mean final `val_bpc` | Mean seconds/run | Mean train tokens/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `64` | 16 | 256000 | 3.0815 | 0.0000 | 3.0374 | 136.9 | 1870.3 |
| `128` | 16 | 512000 | 3.0524 | -0.0291 | 3.0096 | 291.9 | 1755.1 |
| `256` | 8 | 1024000 | 2.9230 | -0.1585 | 2.9088 | 504.5 | 2032.7 |

Interpretation:

- `seq_len=128` was a real but modest win over the old `64` default: `-0.0291` bpc on
  mean final-3 validation.
- `seq_len=256` was decisive under this protocol: `-0.1585` bpc relative to `64` and
  `-0.1294` bpc relative to `128`.
- The `256` win was stable across seeds. Both `seq_len=256` seeds beat both
  `seq_len=128` seeds on mean final-3 `val_bpc`.
- Runtime rose substantially with length, but not prohibitively for this CPU-scale
  matrix. Mean wall time went from `136.9s` at `64` to `504.5s` at `256`.
- Because the budget was fixed in steps instead of tokens, this does not prove the gain
  comes from better long-context exploitation alone. `seq_len=256` also trained on 4x
  as many tokens as `seq_len=64`.

Decision:

- For the current test plan as written, promote `seq_len=256` as the best fixed-step
  sequence-length setting on `greek_classics`.
- Do not make a stronger mechanistic claim than that. A fixed-token control would still
  be needed to separate context-length benefits from raw token-budget benefits.
- Carry the `seq_len=256` checkpoints forward into the post-hoc generation and streaming
  benchmarks before treating the win as fully promoted in writeups.

Artifacts:

- [summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase9_sequence_length_greek_classics_20260423/summary.json)
- [aggregate_summary.json](/Users/peterwei/Desktop/wokspace/reciprocator/runs/lab-book/phase9_sequence_length_greek_classics_20260423/aggregate_summary.json)

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
- Phase 2D's expanded dynamic rank/mode recipe reached `3.1234`, so it does not replace this mode-only recipe.

### Default short-budget optimizer

- `learning_rate=1e-3`
- `lr_warmup_steps=0`
- `lr_decay_style=constant`
- `grad_clip_norm=None`
- `weight_decay=0.0`

This is not just historical inertia. Phase 7 directly tested the proposed
warmup/cosine/clip/decay protocol and found it worse by at least `0.2924` bpc on the
Phase 1 baseline.

## Phase 11: Hybrid Architecture Sanity Scale (Preliminary)

This seed-0 preview was launched on `2026-04-23` / `2026-04-24` and halted
manually before completion. Treat these as optimization-trajectory notes, not as
Phase 11 promotion results. The planned Phase 11 metric remains mean `val_bpc`
over the final 5 eval checkpoints after full 2000-step runs and all seeds.

Run root:

- `runs/lab-book/phase11_hybrid_sanity_greek_classics_parallel_20260423`

Common config:

- `corpus_name=greek_classics`, `max_chars=100000`, train split `90000` chars
- `hidden_size=256`, `state_shape=(4,4,4)`, `coupling_type=sequential`
- `normalization_type=frobenius`
- `token_magnitude_type=inverse_frequency_learned`, `phase_type=rope`, `token_phase=semantic`
- `readout_type=phase_aware`, `enable_self_relation=True`
- Phase 7 optimizer winner: fixed `learning_rate=1e-3`, no warmup, constant LR,
  no gradient clipping, no weight decay
- `seq_len=128`, `batch_size=8`, `eval_every=100`, `eval_batches=16`
- `device=cuda`
- `enable_cross_layer_state=False`
- dynamic mode/rank growth and pruning disabled

Parameter counts from the current code:

| Run | Model | Non-embedding params | Total params |
|---|---:|---:|---:|
| 11A / 11C | 6 Reciprocator layers + 1 attention block (`after`) | 6,382,834 | 6,402,366 |
| 11B | 7 pure Reciprocator layers | 6,810,851 | 6,830,383 |
| 11D | 6 Reciprocator layers + 2 attention blocks (`before`) | 6,909,938 | 6,929,470 |

Latest reported seed-0 values before manual halt:

| Run | Variant | Latest step | Latest `val_bpc` |
|---|---|---:|---:|
| 11A | Hybrid, attention `after`, stateful training | 1000 | 2.5021 |
| 11B | Pure Reciprocator, stateful training | 800 | 2.4671 |
| 11C | Hybrid, attention `after`, stateless training | 1000 | 2.2945 |
| 11D | Hybrid, attention `before`, stateful training | 1000 | 2.5525 |

Preliminary read:

- All four runs learned cleanly from the step-1 `val_bpc` range of roughly `4.8`-`5.1`.
- The stateless hybrid (`11C`) was the clear early leader at the halt point.
- Among stateful runs, the pure baseline (`11B`) was ahead of both hybrid stateful
  variants at its latest reported step, though it had only reported through step `800`.
- This does not settle the architecture question. The small fixed state
  `state_shape=(4,4,4)` may understate the stateful path; follow-up testing should
  include larger fixed state shapes, dynamic mode/rank growth, and
  `enable_cross_layer_state=True`.
- No final checkpoint was written for this preview because the jobs were interrupted
  before step `2000` and were launched without intermediate checkpointing.

## Phase 11x: Post-Growth Cooldown / Larger Batch Preview (Stopped at Step 900)

This preview was launched on `2026-04-24` and stopped manually after all three
runs had written step-900 checkpoints and residual diagnostics. Treat this as a
trajectory/debugging result, not a promotion result: the planned full-budget
metric remains final-5 mean `val_bpc` over completed runs.

Run root:

- `runs/lab-book/phase11x_post_growth_cooldown_bs16_20260424`

Common config:

- `corpus_name=greek_classics`, `hidden_size=256`, `seq_len=128`, `batch_size=16`
- `steps=2000`, `eval_every=100`, `eval_batches=16`, `checkpoint_every=100`
- `num_layers=3`, `attention_every_k=3`, `attention_position=after`
- `token_magnitude_type=inverse_frequency_learned`, `phase_type=rope`
- `readout_type=phase_aware`, `normalization_type=frobenius`
- `enable_self_relation=True`, `device=cuda`
- `learning_rate=1e-3`, `lr_warmup_steps=100`, `lr_decay_style=cosine`
- `min_lr_scale=0.1`, `grad_clip_norm=1.0`, `weight_decay=0.01`
- `record_residual_diagnostics=True`

Run-specific config:

| Run | Stateful | Cross-layer state | Initial state | Dynamic growth |
|---|---:|---:|---:|---:|
| `11C-xlarge-control` | no | no | `(8,12,12)` | no |
| `fixed-after-large-stateful` | yes | yes | `(8,12,12)` | no |
| `dynamic-after-cooldown` | yes | yes | `(4,6,6)` | yes, max `(8,12,12)` |

Dynamic growth settings:

- `dynamic_mode_growth=True`, `dynamic_rank_growth=True`
- `max_rank=10`, `growth_check_interval=50`
- `growth_residual_ema_decay=0.8`
- `min_checks_before_first_growth=7`
- `growth_residual_threshold=0.12`
- `post_growth_cooldown_checks=3`
- `post_growth_cooldown_threshold_scale=1.8`
- `residual_saturate_threshold=0.07`
- `mode_init=mean`

Parameter counts from the current code:

| Run | Non-embedding params | Total params |
|---|---:|---:|
| `11C-xlarge-control` | 11,369,967 | 11,389,499 |
| `fixed-after-large-stateful` | 13,139,441 | 13,158,973 |
| `dynamic-after-cooldown` initial | 3,827,753 | 3,847,285 |

Stopped results:

| Run | Latest step | Latest state | Latest `val_bpc` | Best step | Best `val_bpc` |
|---|---:|---:|---:|---:|---:|
| `11C-xlarge-control` | 900 | `(8,12,12)` | 3.7541 | 200 | 3.4629 |
| `fixed-after-large-stateful` | 900 | `(8,12,12)` | 3.7510 | 300 | 3.4289 |
| `dynamic-after-cooldown` | 900 | `(6,7,6)` | 3.8222 | 200 | 3.4099 |

Validation trajectory:

| Step | 11C control | Fixed large stateful | Dynamic cooldown |
|---:|---:|---:|---:|
| 1 | 6.3389 | 6.2207 | 6.4384 |
| 100 | 3.7166 | 3.7205 | 3.6655 |
| 200 | 3.4629 | 3.5356 | 3.4099 |
| 300 | 3.4761 | 3.4289 | 3.5797 |
| 400 | 3.5424 | 3.5593 | 3.6711 |
| 500 | 3.5108 | 3.4859 | 3.5163 |
| 600 | 3.5653 | 3.6122 | 3.5678 |
| 700 | 3.6238 | 3.7503 | 3.5615 |
| 800 | 3.6987 | 3.7103 | 3.6784 |
| 900 | 3.7541 | 3.7510 | 3.8222 |

Dynamic growth events:

- step `350`: `(4,6,6)` -> `(5,6,6)`
- step `550`: `(5,6,6)` -> `(6,6,6)`
- step `750`: `(6,6,6)` -> `(6,7,6)`

At the stop point, `dynamic-after-cooldown` had
`post_growth_cooldown_checks_remaining=1` and mode residual EMA
`[0.1765, 0.1252, 0.1742]`. The fixed/control runs also had residual EMAs above
or near the base threshold on multiple modes, but no growth path enabled.

Preliminary read:

- The cooldown mechanism prevented runaway growth. Growth occurred every ~200
  steps after eligibility rather than every 50-step check.
- Dynamic was the best run at step `200` and remained competitive through step
  `700`, but the latest growth at step `750` was followed by a clear validation
  regression through step `900`.
- The fixed large stateful model reached the best overall checkpoint in this
  preview (`3.4289` at step `300`) but then degraded. This points more toward
  optimization/schedule instability than a clean capacity win.
- The stateless xlarge control also drifted upward after its step-200 best,
  so the warmup/cosine/clip/decay plus `batch_size=16` protocol is not clearly
  stable on this short run.
- `cross_memory_residual` logged as `0.0` for all runs. Do not interpret that as
  evidence that the state has captured all attention-cache information; this
  diagnostic path still looks inactive or uninformative in the current setup.

## Phase 11Y: Matched Medium Stateful vs Stateless Preview (Bugged Full-State Readout)

This preview was launched on `2026-04-24` with two runs in parallel. Snapshot
recorded at `2026-04-24 05:54:04 UTC`, after both runs had written step-1300
checkpoints and residual diagnostics. The jobs may have continued after this
snapshot.

Run root:

- `runs/lab-book/phase11y_matched_medium_bs16_20260424`

Common config:

- `corpus_name=greek_classics`, `hidden_size=256`, `seq_len=128`, `batch_size=16`
- `steps=2000`, `eval_every=100`, `eval_batches=16`, `checkpoint_every=100`
- `num_layers=3`, `attention_every_k=3`, `attention_position=after`
- `state_shape=(6,8,8)`, `coupling_type=sequential`
- `token_magnitude_type=inverse_frequency_learned`, `phase_type=rope`,
  `token_phase=semantic`
- `readout_type=phase_aware`, `normalization_type=frobenius`
- `enable_self_relation=True`, `device=cuda`
- Phase 7 optimizer winner: `learning_rate=1e-3`, `lr_warmup_steps=0`,
  `lr_decay_style=constant`, `grad_clip_norm=None`, `weight_decay=0.0`
- `record_residual_diagnostics=True`, `track_chunk_drift=False`

Run-specific config:

| Run | Stateful | Cross-layer state | Dynamic growth |
|---|---:|---:|---:|
| `11Y-control-medium-stateless` | no | no | no |
| `11Y-dynamic-medium-to-large` | yes | yes | yes, max `(8,12,12)` |

Dynamic settings:

- `dynamic_mode_growth=True`, `dynamic_rank_growth=True`
- `max_rank=10`, `growth_check_interval=50`
- `growth_residual_ema_decay=0.8`
- `min_checks_before_first_growth=7`
- `growth_residual_threshold=0.12`
- `post_growth_cooldown_checks=3`
- `post_growth_cooldown_threshold_scale=1.8`
- `residual_saturate_threshold=0.07`
- `mode_init=mean`
- pruning disabled in this run

Validation trajectory:

| Step | Control `val_bpc` | Dynamic `val_bpc` | Gap dyn-control | Control acc | Dynamic acc |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.9667 | 4.9618 | -0.0049 | 0.1602 | 0.1587 |
| 100 | 3.3705 | 3.4429 | 0.0724 | 0.3450 | 0.3350 |
| 200 | 2.9002 | 3.0377 | 0.1375 | 0.4221 | 0.4032 |
| 300 | 2.7198 | 2.7849 | 0.0650 | 0.4574 | 0.4411 |
| 400 | 2.5692 | 2.6796 | 0.1104 | 0.4877 | 0.4702 |
| 500 | 2.4364 | 2.6228 | 0.1864 | 0.5209 | 0.4940 |
| 600 | 2.3610 | 2.5046 | 0.1436 | 0.5296 | 0.5016 |
| 700 | 2.3371 | 2.3950 | 0.0580 | 0.5294 | 0.5366 |
| 800 | 2.3182 | 2.4010 | 0.0828 | 0.5422 | 0.5309 |
| 900 | 2.2236 | 2.3864 | 0.1628 | 0.5593 | 0.5412 |
| 1000 | 2.2269 | 2.4196 | 0.1927 | 0.5602 | 0.5324 |
| 1100 | 2.2076 | 2.3892 | 0.1816 | 0.5679 | 0.5358 |
| 1200 | 2.2017 | 2.3605 | 0.1588 | 0.5725 | 0.5536 |
| 1300 | 2.2088 | 2.4111 | 0.2022 | 0.5727 | 0.5477 |

Snapshot results:

| Run | Latest step | Latest state | Latest `val_bpc` | Best step | Best `val_bpc` |
|---|---:|---:|---:|---:|---:|
| `11Y-control-medium-stateless` | 1300 | `(6,8,8)` | 2.2088 | 1200 | 2.2017 |
| `11Y-dynamic-medium-to-large` | 1300 | `(6,8,8)` | 2.4111 | 1200 | 2.3605 |

Dynamic diagnostics at step `1300`:

- `growth_event_history=[]`; no growth fired
- `mode_residual_ema=[0.0399, 0.0246, 0.0237]`, far below the `0.12`
  growth threshold
- `mode_redundancy_ema=[0.1352, 0.1204, 0.1212]`
- cross-layer state was enabled but the learned gates remained small in earlier
  checkpoint inspection (`tanh(beta)` around `-0.0005` and `-0.028` at step `700`)
- `cross_memory_residual=0.0` remained uninformative

Bug discovered during interpretation:

- The Reciprocator mixer currently computes the value returned as `delta` from
  the full `next_state`, not from the incremental state change.
- The relevant path is `ReciprocatorMixer.step`: `delta =
  self.return_map(self._state_features(next_state))`, then
  `ReciprocatorBlock.forward` treats that returned value as the block's delta.
- Therefore the engine is passing a readout of the accumulated recurrent state
  into the hidden stream rather than the intended per-token update signal.

Interpretation under the bug:

- This preview should not be used as evidence against the intended delta-engine
  design. It is a bugged "full-state readout" baseline.
- The observed pattern fits the bug: the stateful/dynamic run fit training harder
  than the stateless control, but validation lagged and became less stable.
- No growth fired because residual pressure fell steadily; under the bug this may
  reflect basis specialization around the accumulated state readout rather than a
  true absence of delta-capacity pressure.
- The stateless control remained the cleaner performer in this bugged preview,
  but the intended Phase 11 comparison needs to be rerun after the mixer returns
  an actual update/delta signal.
- Future rerun should preserve the matched-medium design, but should first fix
  the mixer output and add a regression test that distinguishes full-state readout
  from delta readout.

## Closed, Pending, and Incomplete Branches

Closed by result:

- Spectral backend promotion under the original state shape
- Spectral backend promotion under the higher-dimensional `(8,8,8)` check
- Dynamic spectral backend promotion under Phase 2D (`fft`, `dwt`)
- `virtual_offset` token phase
- Rank-growth promotion in its current form
- Per-mode normalization as a default replacement for Frobenius
- Warmup/cosine scheduling, gradient clipping, and weight decay as defaults for the current 500-step matrix

Incomplete:

- Phase 3.5 full-system dynamic run
- Phase 4 / 4.1 / 4.2 ablations
- Phase 10 scaling branch
- Post-hoc generation and streaming benchmarks as promotion gates

## Recommended Revisions To `docs/test-plan.md`

1. Split the plan from the record. Keep `docs/test-plan.md` strictly prospective and treat this file as the retrospective lab notebook. Right now the plan contains completed results, pending branches, and ad hoc reruns in one place.
2. Add provenance to every run artifact: git commit SHA, run date, config digest, and corpus hash. Without this, comparisons across reruns are weaker than they should be.
3. Update the optimizer protocol section. Phase 7 has now run and rejects the proposed warmup/cosine/clip/decay default for the current 500-step matrix.
4. Tighten the promotion rule. A hard `>0.02` bpc cutoff is close to the scale of some rerun differences; require both a margin and a noise check.
5. Mark the spectral branch as closed. Phase 2S.3 and Phase 2D are now complete and did not clear the promotion rule.
6. Record the current dynamic-mode recipe explicitly in the plan and downgrade it from "candidate winner" to "small-effect exploratory result." The best observed gain is real but below the promotion threshold.
7. Remove rank growth from the active default recipe until the init path is redesigned. Phase 2D made rank growth fire consistently, but the quality result is still neutral-to-negative.
8. Reframe pruning as a debugging task, not a sweep task. Add invariants: pruning must remove exactly one intended axis, preserve rank semantics, and log which axis was pruned. The observed shape collapses need explanation first.
9. Update the run summary table so it matches reality. It should include `phase2_1`, `phase3_2e`, `phase3_2f`, `phase3_4_hi`, and `phase3_4_rerun_20260422`, plus real status markers for completed and closed branches.
10. Make the generation and streaming gates real or mark them pending. The plan says promotion requires them, but the current result summaries do not include those artifacts.
11. Mark blocked prerequisites inline. Phase 4.1 still depends on plumbing `phase_scale` through the CLI/training surface.
12. Reorder the Phase 4 subsections. `4.2` currently appears before `4.1`, which makes the plan harder to scan.
