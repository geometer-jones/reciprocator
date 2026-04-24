# TokenLift + Spectral Coupling Test Plan

## Objective

Identify the best TokenLift configuration across the magnitude, phase, and token-phase
axes. Then screen the recurrent coupling backend across `sequential`, `fft`, `dwt`,
`wavelet_packet`, and `wavelet_packet_max_gauge`. Then test dynamic growth on the best
**sequential** winner only, since dynamic growth and pruning currently support
`coupling_type="sequential"` but not the spectral backends. Because coupling type can
interact with TokenLift choices, treat the spectral phases as a backend screen only:
if a tuned spectral backend wins, rerun the TokenLift axis search and combination under
that backend before declaring the best static no-growth winner. Finally, ablate
specific choices and scalar hyperparameters on that best static winner, including
`phase_scale`, `ffn_expansion_factor`, wavelet depth, and the spectral filter envelope.

## Reporting metric

**Primary:** mean `val_bpc` over the final 3 eval checkpoints of each run.

With `eval_every=50` and `steps=500`, the final 3 evals land at steps 400, 450, 500.
Averaging smooths out single-step noise without discarding early convergence signal.

Winner selection uses the mean across seeds: `mean(val_bpc_final3, seeds=[0,1,2])`.

**Secondary optimization metrics:** report the optimizer protocol used for each run
(`lr`, `lr_warmup_steps`, `lr_decay_style`, `min_lr_scale`, `grad_clip_norm`,
`weight_decay`) so later comparisons are not confounded by silent training-method
changes.

**Secondary generation metrics:** for every promoted checkpoint, emit sampled text plus
`distinct-1` and `distinct-2` on the continuation. Perplexity-only wins do not count as
clean wins if they collapse into repetition or unreadable samples.

**Secondary systems metrics:** for every promoted checkpoint, benchmark streaming
inference wall-clock and memory across prompt lengths. The Reciprocator claims O(1)
decode-state usage; the plan must check that directly rather than infer it from loss.

---

## Fixed hyperparameters

| Parameter             | Value    | Why / change from original                                        |
|-----------------------|----------|-------------------------------------------------------------------|
| corpus                | greek_classics |                                                             |
| max_chars             | 100,000  | 20k gives ~280 windows; 100k gives ~1,400 — enough for signal    |
| val_fraction          | 0.1      |                                                                   |
| batch_size            | 8        |                                                                   |
| seq_len               | 64       |                                                                   |
| learning_rate         | 1e-3     | 3e-3 is high for complex-valued params; 1e-3 is safer            |
| hidden_size           | 32       |                                                                   |
| phase_scale           | π        | TensorSignalProjector default; varied in Phase 4.1               |
| ffn_expansion_factor  | 2        | Standard capacity knob; varied in Phase 4.1                      |
| readout_type          | phase_aware | Phase 4 promoted over `magnitude`; historical phases before Phase 4 used `magnitude` |
| state_shape           | (2,3,4)  | phases 1–2                                                       |
| coupling_type         | sequential | default for phases 0–2 and Phase 3; screened statically in Phase 2S and dynamically in Phase 2D |
| low_frequency_gain    | 0.5      | spectral screen default when `coupling_type != sequential`; varied in Phase 2S.2 |
| low_frequency_sigma   | 0.35     | spectral screen default when `coupling_type != sequential`; varied in Phase 2S.2 |
| high_frequency_gain   | 0.5      | spectral screen default when `coupling_type != sequential`; varied in Phase 2S.2 |
| high_frequency_cutoff | 0.5      | spectral screen default when `coupling_type != sequential`; varied in Phase 2S.2 |
| dynamic_spectral_gains | False   | adaptive spectral envelope flag; enable only for explicit learned-filter ablations |
| anisotropic_spectral_gains | False | FFT dynamic-gain opt-in for full coordinatewise frequency-grid modulation |
| wavelet_levels        | None     | spectral screen default; wavelet depth swept in Phase 2S.1      |
| dynamic_mode_growth    | False   | phases 1–2                                                       |
| dynamic_rank_growth    | False   | phases 1–2                                                       |
| enable_self_relation  | False    |                                                                   |
| num_layers            | 1        | phases 0–3.5; run 16 probes static depth and Phase 4.2 conditionally validates multi-layer growth |
| eval_every            | 50       | default 20 is too frequent; 50 keeps 10 eval points per 500 steps |
| eval_batches          | 16       | default 4 = 2k tokens per eval (noisy); 16 = 8k tokens          |
| seeds                 | [0, 1, 2] | phases 1–2; phases 3–4 use [0, 1] (see note)                   |
| steps                 | 500      | phases 1–2                                                       |

**Phase 3 and 4 seeds:** using 2 seeds ([0, 1]) rather than 3. Phase 1–2 establish
the winner; phases 3–4 are probing growth mechanics and specific ablations. The
computational cost of 3 seeds on 8 runs with 1000 steps each is high. If resources
allow, extend to 3.

## Optimization protocol (to validate in Phase 7)

Treat the following as the current proposed default for all future phases after Phase 7
lands a winner:

| Parameter        | Proposed value | Why                                                            |
|------------------|----------------|----------------------------------------------------------------|
| learning_rate    | 1e-3           | Preserves current scale while separating schedule from raw lr   |
| lr_warmup_steps  | 50             | Avoids shocking complex-valued / normalized dynamics at step 1  |
| lr_decay_style   | cosine         | Converts the fixed-lr pilot into a finite-budget schedule       |
| min_lr_scale     | 0.1            | Lets training keep refining late instead of flat-lining at full lr |
| grad_clip_norm   | 1.0            | Guards against occasional exploding updates                     |
| weight_decay     | 0.01           | Mild regularization on a small, highly-coupled model           |

Until Phase 7 is run, Phases 0–6 remain historically valid but legacy: keep their
results, then use the Phase 7 winner as the optimization protocol for all new runs.

## Evaluation add-ons

These are not separate model families; they are attached evaluations for promoted
checkpoints.

| Evaluation              | Value            | Notes                                                      |
|------------------------|------------------|------------------------------------------------------------|
| generation_eval_samples | 4               | Enough to spot repetition/collapse without expensive review |
| generation_prompt_len   | 64              | Long enough to anchor style and syntax                     |
| generation_new_tokens   | 256             | Forces the model to sustain generation beyond a short stub |
| generation_temperature  | 0.8             | Moderately exploratory, not pure argmax                    |
| generation_top_k        | 20              | Prevents heavy-tail garbage while preserving diversity     |
| benchmark_prompt_lengths | (32,128,512,1024) | Direct prompt-length sweep for the O(1) decode claim    |
| benchmark_new_tokens    | 256             | Long enough to smooth token-level timing noise             |

---

## Phase 0: Pilot ✓ COMPLETE

**Results (seed=0, 500 steps):**
- Stability: passed. No NaN/Inf.
- Final `val_bpc` at step 500: 4.0774.
- Final-3 mean (steps 400/450/500): 4.1453.
- Wall time: 61.64s on CPU.
- Dataset: train_tokens=90,000, val_tokens=10,000.

**Notes:** val_bpc landed above the original target band (2.0–3.5). The curve was still
improving materially at step 500, indicating the budget is training-limited rather than
capacity-limited. Acceptance band revised to 3.5–5.5. This is now treated as evidence
against a fixed learning rate: future runs should use Phase 7 to validate warmup +
decay before extending the rest of the matrix. For Phase 3, keep
an EMA-smoothed mode residual trigger (`growth_residual_threshold`, `growth_residual_ema_decay`) so
mode growth fires on a stable structural-gap signal rather than a single-batch spike.

---

## Phase 1: Isolate each TokenLift axis ✓ COMPLETE

**18 runs** (6 configs × 3 seeds). Steps = 500. No growth.

| #  | token_magnitude_type      | phase_type  | token_phase | What it tests           |
|----|---------------------------|-------------|-------------|-------------------------|
| 1  | learned                   | rope        | none        | Baseline                |
| 2  | inverse_frequency_learned | rope        | none        | Magnitude axis          |
| 3  | learned                   | locked_wave | none        | Full phase coupling     |
| 4  | learned                   | local_wave  | none        | Local phase coupling    |
| 5  | learned                   | rope        | semantic    | Semantic token phase    |
| 6  | learned                   | rope        | virtual_offset | Virtual position offset |

**Decision rule for Phase 2 (specify before running):**

For each axis, compute `Δbpc = mean_bpc(variant) - mean_bpc(run_1)`. A variant is
selected for Phase 2 only if `Δbpc < -0.02` (i.e., improves by more than 0.02 bpc
over baseline). If no variant on an axis beats baseline by that margin, keep the
baseline default for that axis.

- Magnitude axis: compare run 2 vs run 1.
- Phase axis: compare runs 3–4 vs run 1; pick the better of the two if either qualifies.
- Token-phase axis: compare runs 5–6 vs run 1; pick the better of the two if either qualifies.

This rule must be applied before looking at Phase 2 results.

---

## Phase 2: Combine Phase 1 winners ✓ COMPLETE

**Up to 9 runs** (up to 3 configs × 3 seeds). Steps = 500. No growth.

Substitute the Phase 1 winners into each axis. The three runs are:

| #  | token_magnitude_type       | phase_type           | token_phase           | Rationale                       |
|----|----------------------------|----------------------|-----------------------|---------------------------------|
| 7  | [mag winner or learned]    | [phase winner or rope] | none                | Best magnitude + best phase     |
| 8  | [mag winner or learned]    | rope                 | [token_phase winner or none] | Best magnitude + token phase |
| 9  | [mag winner or learned]    | [phase winner or rope] | [token_phase winner or none] | Full stack                 |

If a given axis showed no Phase 1 winner, the baseline value for that axis collapses
some of these runs to duplicates — skip duplicates.

**Decision rule for Phase 2S, Phases 2T–2U, and Phase 3:** The Phase 2 config with the
lowest mean `val_bpc` (across seeds, final-3 average) is the "Phase 2 sequential
winner" and becomes the fixed config for the spectral screen in Phase 2S and the
dynamic-growth branch in Phases 3–3.5. Record the exact hyperparameters before moving
on. If the tuned spectral candidate later wins, do not assume this sequential winner
remains the best static config under that backend; run Phases 2T–2U first.

---

## Phase 2S: Spectral coupling screen

**15 runs** (5 configs × 3 seeds). Steps = 500. No growth. Uses the Phase 2 sequential
winner TokenLift stack as a fixed base.

This phase isolates the new recurrent coupling choices without mixing them into the
TokenLift axis search. The `sequential` run is the direct baseline; the spectral runs
keep the shared filter envelope fixed at the defaults listed above. This is a backend
screen, not a full cross with TokenLift axes: a spectral win here means "promising
backend on the Phase 2 sequential winner," not "the sequential TokenLift ranking
transfers unchanged."

| Run  | coupling_type               | What it tests                                                |
|------|-----------------------------|--------------------------------------------------------------|
| 2S1  | sequential                  | Existing mixer baseline                                      |
| 2S2  | fft                         | N-dimensional FFT over tensor axes with smooth radial filter |
| 2S3  | dwt                         | Flattened 1-D wavelet pyramid                                |
| 2S4  | wavelet_packet              | Full packet tree with entropy best-basis                     |
| 2S5  | wavelet_packet_max_gauge    | Packet tree with entropy + phase-coherence best-basis        |

**Decision rule for calibration follow-ups:** compare each spectral run against
run 2S1. If no non-sequential backend beats `sequential` by > 0.02 bpc, skip Phases
2S.1, 2S.2, 2T, and 2U and keep the Phase 2 sequential winner as the best static
winner. Otherwise carry the best wavelet-family backend forward into Phase 2S.1, and
carry the strongest non-sequential spectral candidate forward into Phase 2S.2. Do not
finalize the best static winner yet; that decision is made only after depth and filter
calibration in Phase 2S.2.

**Continuity constraint:** even if a spectral backend wins this phase, Phases 3–3.5
stay on the Phase 2 sequential winner so the original dynamic-growth baseline remains
comparable. Phase 2D is the dynamic backend screen for sequential and spectral
couplings.

---

## Phase 2S.1: Wavelet depth calibration

**Conditional phase:** run only if Phase 2S finds a non-sequential backend that beats
`sequential` by more than 0.02 bpc.

**9 runs** (3 configs × 3 seeds). Steps = 500. No growth. Uses the best wavelet-family
backend from Phase 2S (`dwt`, `wavelet_packet`, or `wavelet_packet_max_gauge`) as the
fixed base.

Purpose: determine whether wavelet-family performance is limited by using full-depth
decomposition (`wavelet_levels=None`) rather than the family itself.

The inherited Phase 2S run at `wavelet_levels=None` is the baseline; this phase adds
three shallower depths:

| Run   | wavelet_levels | What it tests                   |
|-------|----------------|---------------------------------|
| 2S.1a | 1              | Very shallow decomposition      |
| 2S.1b | 2              | Moderate decomposition          |
| 2S.1c | 3              | Deeper but still not full-depth |

**Decision rule:** identify the best wavelet-depth setting by comparing each run
against the inherited full-depth baseline for the same backend. If none of the shallow
depths improves by > 0.02 bpc, keep `wavelet_levels=None` as the wavelet default.

---

## Phase 2S.2: Spectral filter calibration

**Conditional phase:** run only if Phase 2S finds a non-sequential backend that beats
`sequential` by more than 0.02 bpc.

**9 runs** (3 configs × 3 seeds). Steps = 500. No growth. Uses the strongest
non-sequential spectral candidate as the fixed base:
- `fft` from Phase 2S, if it outperforms all wavelet-family results
- otherwise the best wavelet-family result after Phase 2S.1 depth calibration

Purpose: commit an actual test of the four spectral filter hyperparameters without
exploding into a full continuous sweep. The plan uses three coupled presets that span
gentle, default-adjacent, and aggressive filtering regimes.

The inherited default spectral run (`low_frequency_gain=0.5`, `low_frequency_sigma=0.35`,
`high_frequency_gain=0.5`, `high_frequency_cutoff=0.5`) is the comparison baseline; this
phase adds:

| Run   | low_frequency_gain | low_frequency_sigma | high_frequency_gain | high_frequency_cutoff | What it tests                                    |
|-------|--------------------|---------------------|---------------------|-----------------------|--------------------------------------------------|
| 2S.2a | 0.25               | 0.50                | 0.25                | 0.65                  | Gentle smoothing, late high-frequency damping    |
| 2S.2b | 0.75               | 0.20                | 0.50                | 0.65                  | Strong low-frequency emphasis, permissive cutoff |
| 2S.2c | 0.75               | 0.20                | 0.75                | 0.35                  | Aggressive low-pass / hard damping               |

**Decision rule for the tuned spectral candidate:** compare the best tuned spectral
candidate (after Phase 2S.1 and Phase 2S.2) against Phase 2S run 2S1 (`sequential`).
If the tuned spectral candidate beats `sequential` by > 0.02 bpc, carry it forward into
Phases 2T and 2U. Otherwise skip Phases 2T and 2U and keep the Phase 2 sequential
winner as the best static winner for Phases 4 and 9.

---

## Phase 2S.3: Spectral screen at higher state dimensionality

**Conditional phase:** run only if Phase 2S finds no non-sequential backend that beats `sequential` by more than 0.02 bpc.

**15 runs** (5 configs × 3 seeds). Steps = 500. No growth. Same structure as Phase 2S but with `state_shape=(8,8,8)`.

**Motivation:** at `state_shape=(2,3,4)`, the largest mode has 4 elements — wavelet decomposition produces at most 2 levels and FFT has 3 non-trivial frequency bins. This phase tests whether the null result was a dimensionality artifact by giving spectral backends a tensor large enough to exhibit meaningful frequency structure (≥3 wavelet levels per mode, 5 non-trivial FFT bins per axis).

| Run   | coupling_type               | What it tests                                                |
|-------|-----------------------------|--------------------------------------------------------------|
| 2S.3a | sequential                  | Sequential baseline at `(8,8,8)`                            |
| 2S.3b | fft                         | FFT over 512-element tensor with radial filter               |
| 2S.3c | dwt                         | Wavelet pyramid with ≥3 meaningful levels per mode           |
| 2S.3d | wavelet_packet              | Packet tree with sufficient leaves for entropy discrimination |
| 2S.3e | wavelet_packet_max_gauge    | Packet tree with entropy + phase-coherence best-basis        |

**Decision rule:** compare each spectral run against 2S.3a. If no spectral backend beats sequential by > 0.02 bpc: the Phase 2S null is confirmed — sequential coupling is the correct inductive bias for this architecture regardless of state dimensionality; close the spectral investigation. If a spectral backend wins: the Phase 2S null was a dimensionality artifact; further depth and filter calibration at `state_shape=(8,8,8)` is warranted before adopting spectral as a default.

---

## Phase 2S.4: FFT learned/anisotropic gain ablation

**Follow-up phase:** run after Phase 2S.3 when testing the isotropic-filter critique
directly. This is a focused FFT ablation, not a renewed wavelet screen.

**15 runs** (5 configs × 3 seeds). Steps = 500. No growth. Uses the Phase 2 sequential
winner TokenLift stack and `state_shape=(8,8,8)` so the FFT grid has enough frequency
coordinates for anisotropy to matter.

Keep the fixed spectral envelope at the defaults:
`low_frequency_gain=0.5`, `low_frequency_sigma=0.35`,
`high_frequency_gain=0.5`, `high_frequency_cutoff=0.5`.

| Run    | coupling_type | dynamic_spectral_gains | anisotropic_spectral_gains | What it tests |
|--------|---------------|------------------------|----------------------------|---------------|
| 2S.4a | sequential    | False                  | False                      | Direct recurrent baseline |
| 2S.4b | fft           | False                  | False                      | Fixed radial FFT baseline |
| 2S.4c | fft           | True                   | False                      | Learned radial FFT gains |
| 2S.4d | fft           | False                  | True                       | Inert anisotropic-flag control; should match 2S.4b |
| 2S.4e | fft           | True                   | True                       | Learned full coordinatewise anisotropic FFT gains |

**Decision rule:** first confirm 2S.4d matches 2S.4b within same-seed noise; any
meaningful difference means the anisotropic flag is affecting fixed FFT and should be
treated as a bug. Then compare 2S.4c against 2S.4b to measure the value of learned
radial gains, and 2S.4e against 2S.4c to isolate anisotropy. Promote the anisotropic
FFT path only if 2S.4e beats both 2S.4a and 2S.4c by > 0.02 bpc on final5 `val_bpc`.

---

## Phase 2D: Dynamic rank/mode backend adaptation screen

**Follow-up phase:** run after Phase 2S.4 if we want to test expanded tensor capacity
with dynamic adaptation rather than another static spectral backend screen.

**9 runs** (3 configs × 3 seeds). Steps = 1000. Uses the Phase 2 sequential winner
TokenLift stack.

This phase explicitly checks dynamic adaptation across `sequential`, `fft`, and `dwt`.
Both `wavelet_packet` and `wavelet_packet_max_gauge` are excluded by request.

**State shape:** start from `state_shape=(4,8,8,8)`.

This expands both:
- rank: `3 -> 4`
- state elements: `512 -> 2048` relative to Phase 2S.3's `(8,8,8)` screen

**Dynamic settings for all 2D runs:**

| Parameter | Value |
|---|---|
| dynamic_mode_growth | True |
| dynamic_rank_growth | True |
| dynamic_mode_pruning | True |
| dynamic_rank_pruning | True |
| max_state_shape | `(6,10,10,10)` |
| max_rank | 5 |
| growth_check_interval | 50 |
| growth_residual_ema_decay | 0.8 |
| min_checks_before_first_growth | 7 |
| growth_residual_threshold | 0.12 |
| residual_saturate_threshold | 0.07 |
| prune_threshold | 0.03 |
| prune_sustain_steps | 3 |
| prune_min_steps | 150 |
| mode_init | mean |
| rank_init | zero |
| rank_growth_loss_ceiling | 1.5 |

Pruning is deliberately conservative because Phase 3.4 showed that aggressive pruning
can damage performance. This phase still enables pruning so spectral dynamic adaptation
is tested end-to-end, but promotion requires non-pathological state-shape event logs.

| Run | Coupling | What it tests |
|-----|----------|---------------|
| 2D.1 | sequential | Dynamic rank/mode adaptation control |
| 2D.2 | fft | Fourier spectral coupling under dynamic adaptation |
| 2D.3 | dwt | Haar DWT spectral coupling under dynamic adaptation |

**Reporting metric:** mean `val_bpc` over the final 5 eval checkpoints
(steps 800, 850, 900, 950, 1000).

**Reporting:** include full state-shape event logs, final `state_shape`, whether mode
growth fired, whether rank growth fired, whether mode/rank pruning fired, and residual
diagnostics for every dynamic run.

**Decision rule:** compare spectral dynamic runs against 2D.1. Promote a spectral
dynamic backend only if it improves by > 0.02 bpc, rank growth fires in at least one
seed, and pruning does not cause obvious grow/prune churn or an early quality collapse.
If rank growth never fires, report the result as "mode-only in practice" and do not
claim rank adaptation worked.

---

## Phase 2T: TokenLift axis transfer check under the tuned spectral backend

**Conditional phase:** run only if the tuned spectral candidate from Phase 2S.2 beats
`sequential` by more than 0.02 bpc.

**18 runs** (6 configs × 3 seeds). Steps = 500. No growth. Fix `coupling_type` and all
spectral filter settings to the tuned spectral candidate selected by Phases 2S–2S.2.

This mirrors Phase 1 because coupling type can change the TokenLift landscape. The
sequential rankings from Phases 1–2 are not assumed to transfer.

| Run  | token_magnitude_type      | phase_type  | token_phase     | What it tests                                |
|------|---------------------------|-------------|-----------------|----------------------------------------------|
| 2T1  | learned                   | rope        | none            | Spectral baseline                            |
| 2T2  | inverse_frequency_learned | rope        | none            | Magnitude axis under tuned spectral backend  |
| 2T3  | learned                   | locked_wave | none            | Full phase coupling under tuned spectral     |
| 2T4  | learned                   | local_wave  | none            | Local phase coupling under tuned spectral    |
| 2T5  | learned                   | rope        | semantic        | Semantic token phase under tuned spectral    |
| 2T6  | learned                   | rope        | virtual_offset  | Virtual position offset under tuned spectral |

**Decision rule:** identical to Phase 1, but applied against run 2T1. A variant is
selected for Phase 2U only if it improves over run 2T1 by more than 0.02 bpc on the
final-3 mean across seeds.

---

## Phase 2U: Combine tuned spectral-axis winners

**Conditional phase:** run only if Phase 2T runs.

**Up to 9 runs** (up to 3 configs × 3 seeds). Steps = 500. No growth. Uses the
Phase 2T axis winners under the tuned spectral backend from Phases 2S–2S.2.

| Run  | token_magnitude_type          | phase_type                | token_phase                     | Rationale                           |
|------|-------------------------------|---------------------------|---------------------------------|-------------------------------------|
| 2U1  | [2T mag winner or learned]   | [2T phase winner or rope] | none                            | Best magnitude + best phase         |
| 2U2  | [2T mag winner or learned]   | rope                      | [2T token_phase winner or none] | Best magnitude + token phase        |
| 2U3  | [2T mag winner or learned]   | [2T phase winner or rope] | [2T token_phase winner or none] | Full tuned spectral TokenLift stack |

Skip duplicates where axes collapsed to baseline.

**Decision rule:** the Phase 2U config with the lowest mean `val_bpc` becomes the best
static winner for Phases 4 and 9. If all axes collapse to baseline, the best static
winner is simply the tuned spectral baseline under the Phase 2S-selected backend.

---

## Phase 3: Static baseline

**2 runs** (1 config × 2 seeds). Steps = 1000. Uses the Phase 2 sequential winner
config. No growth.

Essential control: separates "dynamic growth helps" from "more training steps help."
All Phase 3.x growth results are compared against this.

| #  | dynamic_mode_growth | dynamic_rank_growth | What it tests          |
|----|---------------------|---------------------|------------------------|
| 10 | False                   | False                   | Static at 1000 steps |

**Reporting metric:** mean `val_bpc` over the final 5 eval checkpoints (steps 800, 850,
900, 950, 1000).

---

## Phase 3.1: Residual EMA calibration

**6 runs** (3 configs × 2 seeds). Steps = 1000. No growth. Uses the Phase 2 sequential
winner config.

**Purpose:** establish the typical magnitude and trajectory of per-mode EMA residual
norms over training, so that growth and pruning thresholds in Phases 3.2–3.4 can be
chosen from data rather than guessed.

**Mechanism:** at every growth/prune check (every `growth_check_interval=50` steps), compute the
per-mode QR residual — project the current reference context signal (from
`_collect_growth_reference_contexts`) against the column space of existing slices along
each mode. Update a running EMA per mode:

```
ema_residual[m] = alpha * residual_norm[m] + (1 - alpha) * ema_residual[m]
```

In code this is controlled by `growth_residual_ema_decay`, with
`residual_ema_alpha = 1 - growth_residual_ema_decay`. So `alpha=0.2` means
`growth_residual_ema_decay=0.8`.

Also track a per-mode redundancy EMA for each mode k against the union of all other
modes — this will be needed to set pruning thresholds in Phase 3.4.

No growth fires. EMA values are logged at every check for post-hoc analysis.

**What to vary:** `growth_residual_ema_decay` in code, expressed below alongside the
equivalent `residual_ema_alpha = 1 - decay`. At `growth_check_interval=50`, the
effective window in training steps is approximately `growth_check_interval / alpha`:

| Run  | residual_ema_alpha | growth_residual_ema_decay | Effective window (steps) |
|------|--------------------|---------------------------|--------------------------|
| 3.1a | 0.1                | 0.9                       | ~500                     |
| 3.1b | 0.2                | 0.8                       | ~250                     |
| 3.1c | 0.3                | 0.7                       | ~165                     |

**Reporting:** per-mode EMA residual curves and per-mode redundancy curves for all
three runs, both seeds. Record:
- Typical residual magnitude at steps 50, 200, 500, 1000
- Rate of decay — does each mode plateau, and at what value?
- Whether modes saturate at different rates
- Typical redundancy levels at convergence

**Decision rule (apply before Phase 3.2):** choose `growth_residual_ema_decay`
(equivalently `residual_ema_alpha = 1 - decay`), and read off:
- `residual_grow_threshold` — the residual level where curves are still meaningfully
  above their eventual plateau (modes have room to grow)
- `residual_saturate_threshold` — just above the noise floor at convergence (modes are
  structurally exhausted)
- `prune_threshold` — the redundancy level at which a mode adds negligible new signal

---

## Phase 3.2: Mode growth with residual EMA trigger

**10 runs** (5 configs × 2 seeds). Steps = 1000. Mode growth only
(`dynamic_rank_growth=False`). Uses the Phase 2 sequential winner config and
`growth_residual_ema_decay` from Phase 3.1.

**Trigger:** at each growth check, grow mode m if `ema_residual[m] >
residual_grow_threshold`. Grow the mode with the highest EMA residual. No
additional loss-floor gate.

**Init:** orthogonal-seeded — the new slice is initialized to the QR orthogonal
complement of the EMA residual direction that triggered growth. This is the direction
existing slices cannot represent, computed from the signal the model has been
accumulating. The slice is immediately active after insertion.

**Fixed params:**

| Parameter                | Value                       | Notes                                                                 |
|--------------------------|-----------------------------|-----------------------------------------------------------------------|
| growth_check_interval    | 50                          | Shared cadence for both growth and prune checks                       |
| growth_residual_ema_decay | [Phase 3.1 winner]         | Equivalent to `residual_ema_alpha = 1 - decay`                        |
| min_checks_before_first_growth | [Phase 3.1 winner]    | EMA updates required before growth is eligible; see note below        |
| max_state_shape          | (6,6,6)                     |                                                                       |

**`min_checks_before_first_growth`:** at the start of training the EMA has no history — the first
few values reflect initialization noise rather than structural signal. Growth is
suppressed until at least `min_checks_before_first_growth` EMA updates have been recorded. The
right value is roughly 1–2 EMA time constants (`1 / residual_ema_alpha` to
`2 / residual_ema_alpha` checks, where `residual_ema_alpha = 1 - growth_residual_ema_decay`).
At `alpha=0.2` / `decay=0.8` that is 5–10 checks (250–500 steps); at
`alpha=0.3` / `decay=0.7` it is 3–7 checks (150–350 steps). Read the inflection point from the
Phase 3.1 residual curves — growth should become eligible once the per-mode curves have
stopped their initial rapid descent and entered the slower plateau phase.

**Sub-experiment A — threshold sensitivity** (verify Phase 3.1 calibration fires
at sensible times, not at step 50):

| Run  | residual_grow_threshold     | What it tests               |
|------|-----------------------------|-----------------------------|
| 3.2a | [calibrated × 0.7]          | More aggressive trigger     |
| 3.2b | [calibrated]                | Calibrated threshold        |
| 3.2c | [calibrated × 1.4]          | More conservative trigger   |

If growth still fires at step 50 across all runs, `residual_grow_threshold` is too low
— raise it. If growth never fires, lower it.

**Sub-experiment B — init strategy** (using Phase 3.2a–c winner threshold):

| Run  | mode_init | What it tests                                              |
|------|-----------|------------------------------------------------------------|
| 3.2d | zero      | Inert baseline                                             |
| 3.2e | orthogonal | QR orthogonal complement of the EMA residual direction — targets the gap |

Mean init is not tested: the EMA residual direction is strictly more informative (mean
of existing slices points at what current capacity already does, not what it can't do).
`mode_init=residual` remains a separate raw-residual scaling strategy in code, distinct
from the QR-orthogonal `mode_init=orthogonal` path described here.

**Reporting:** per-mode EMA residual traces, growth event log (when, which mode, new
shape), final5 val_bpc. Compare against Phase 3 run 10.

**Decision rule:** adopt threshold from 3.2a–c where growth timing is sensible (not
step 50, not never). Adopt residual init over zero if improvement > 0.02 bpc. Carry
forward winners as defaults for Phase 3.3+.

---

## Phase 3.3: Rank growth with residual EMA trigger

**6 runs** (3 configs × 2 seeds). Steps = 1000. Rank growth only
(`dynamic_mode_growth=False`). Uses the Phase 2 sequential winner config and Phase 3.2
residual EMA params.

**Trigger:** rank growth fires when both conditions hold simultaneously:
`all(ema_residual[m] < residual_saturate_threshold for m in modes)` — all modes are
structurally saturated (no mode has a gap worth filling).

**Init:** residual-seeded — at trigger time the full reconstruction residual (the
component of the hidden signal not explained by the current tensor forward pass) seeds
the new mode. Unlike mode growth where the residual is per-mode, rank init uses the
cross-mode residual.

**Fixed params:**

| Parameter                    | Value                         | Notes                                          |
|------------------------------|-------------------------------|------------------------------------------------|
| growth_check_interval        | 50                            | Shared cadence for both growth and prune checks |
| growth_residual_ema_decay    | [Phase 3.1 winner]            | Equivalent to `residual_ema_alpha = 1 - decay` |
| min_checks_before_first_growth | [Phase 3.1 winner]          | Same guard as Phase 3.2; rank growth also suppressed until EMA has history |
| residual_saturate_threshold  | [Phase 3.1 calibration]       |                                                |
| max_rank                     | 5                             |                                                |

| Run  | rank_init | What it tests                                                    |
|------|-----------|------------------------------------------------------------------|
| 3.3a | zero      | Inert baseline                                                   |
| 3.3b | residual  | Cross-mode reconstruction residual at trigger time               |
| 3.3c | mean      | Simple low-shock control between zero and residual               |

**Reporting:** growth event log, step at which rank growth first fires, final
state_shape, final5 val_bpc. Key check: does rank growth now fire *after* modes have
saturated rather than at step 50?

**Decision rule:** adopt a non-zero init only if it improves over zero by > 0.02 bpc.
Use mean init as a stability control: if residual underperforms both zero and mean, the
cross-mode residual seed is too disruptive in its current form. If rank growth never
fires, `rank_growth_loss_ceiling` is too low — raise it and re-run. If rank growth
fires at step 50, `residual_saturate_threshold` is too high — lower it.

---

## Phase 3.4: Pruning

**6 runs** (3 configs × 2 seeds). Steps = 1000. Mode growth enabled (Phase 3.2
winner config). `dynamic_rank_growth=False`.

**Trigger:** at each growth check, mark mode k as a pruning candidate if
`ema_redundancy[k] < prune_threshold`. Prune mode k if it has remained a candidate for
`prune_sustain_steps` and `steps_since_last_growth[k] > prune_min_steps`. The sustain
window and cooldown guard against grow-prune oscillation.

**Fixed params:**

| Parameter          | Value                    | Notes                                          |
|--------------------|--------------------------|------------------------------------------------|
| mode growth config | [Phase 3.2 winner]       |                                                |
| prune_min_steps    | 200                      | Cooldown after last growth event for that mode |
| prune_threshold    | [Phase 3.1 calibration]  |                                                |

**Threshold and hysteresis calibration:**

| Run  | prune_threshold    | prune_sustain_steps | What it tests                         |
|------|--------------------|---------------------|---------------------------------------|
| 3.4a | [calibrated]       | 100                 | Moderate threshold, tight hysteresis  |
| 3.4b | [calibrated]       | 200                 | Moderate threshold, loose hysteresis  |
| 3.4c | [calibrated × 0.7] | 200                 | Conservative — only obvious redundancy |

**Reporting:** prune event log (when, which mode, shape before/after), net growth event
count (grows minus prunes), final state_shape, final5 val_bpc vs Phase 3.2 winner
without pruning.

**Decision rule:** adopt pruning config if val_bpc matches Phase 3.2 winner within 0.02
with fewer net parameters. If pruning consistently degrades performance, threshold is
too aggressive — raise it. If nothing is ever pruned, threshold is too conservative —
lower it.

---

## Phase 3.5: Full system

**4 runs** (2 configs × 2 seeds). Steps = 1000. Uses the Phase 2 sequential winner
config and all Phase 3.2–3.4 winners.

| Run  | mode_growth | rank_growth | pruning | What it tests              |
|------|-------------|-------------|---------|----------------------------|
| 3.5a | True        | False       | True    | Mode growth + pruning      |
| 3.5b | True        | True        | True    | Full system                |

**Comparison:** run 3.5b against Phase 3 run 10 (static baseline) and Phase 3.2 winner
(mode growth only, no rank, no pruning). This is the definitive test of whether the
full residual-EMA-driven system outperforms both static training and the simpler
growth-only approach.

**Reporting:** full event log (grows and prunes interleaved), net mode count trajectory
over training, final5 val_bpc.

**Scope note:** all Phase 3.x growth results are single-layer only (`num_layers=1`).
Growth and pruning decisions currently aggregate per-layer residual norms into one
trigger signal, so a static depth win must not be assumed to transfer to dynamic
growth. Phase 4.2 is the explicit validation gate for multi-layer growth.


## Phase 4: Ablations on the best static winner

**Up to 12 runs** (up to 6 configs × 2 seeds). Steps = 500. No growth. Uses the best
static winner as the base, varying one thing at a time. If spectral calibration finds
no improvement, this is the Phase 2 sequential winner. If a tuned spectral backend
wins, this is the Phase 2U winner under that backend; keep the finalized spectral
backend and filter settings fixed during all Phase 4 ablations.

| #  | Variation                                              | Baseline for comparison | What it tests                                       |
|----|--------------------------------------------------------|-------------------------|-----------------------------------------------------|
| 14 | token_magnitude_type = inverse_frequency               | Best static winner      | Is the learned residual on magnitude helping?        |
| 15 | readout_type = phase_aware                             | Best static winner      | Does preserving phase in readout help?               |
| 16 | num_layers = 2                                         | Best static winner      | Does depth help?                                    |
| 17 | token_phase = semantic_virtual_offset                  | Best static winner      | Does virtual_offset add anything on top of semantic? |
| 18 | enable_self_relation = True                            | Best static winner      | Does the prior-state × present Hadamard term help?   |

**Notes on conditionals:**

- Run 14 only makes sense if the best static winner uses `inverse_frequency_learned`.
  If it uses `learned`, skip run 14 (no ablation to do).
- Run 16 is a **static-only** depth check. If it wins, do not automatically promote
  multi-layer growth; use Phase 4.2 to validate whether the layer-aggregated growth
  trigger still behaves sensibly when `num_layers > 1`.
- Run 17 only makes sense if the best static winner uses `token_phase=semantic`. If the
  winner uses `none` or `virtual_offset`, reformulate or skip.
---

## Phase 4.2: Multi-layer growth sanity check

**Conditional:** run because Phase 4C showed that `readout_type=phase_aware` and
`num_layers=2` compose strongly, improving over the single-layer phase-aware recipe by
`0.0691` bpc.

**4 runs** (2 configs × 2 seeds). Steps = 1000. Sequential coupling only. Uses the
Phase 4C static quality recipe:

- `readout_type=phase_aware`
- `num_layers=2`
- `token_magnitude_type=inverse_frequency_learned`
- `phase_type=rope`
- `token_phase=semantic`
- `normalization_type=frobenius`

The dynamic arm transfers only the best single-layer mode-growth recipe from Phase
3.2f. Do not enable rank growth or pruning here: rank growth and pruning are not
promoted, and this phase is isolating whether mode-growth triggers remain interpretable
with multiple layers.

**Purpose:** validate whether EMA-driven growth/pruning still works when residual norms
are aggregated across multiple layers. Today `_compute_mode_residual_norms` and the
paired pruning residual path collapse per-layer signals into a single per-mode trigger.
That may be fine, or it may hide layer disagreement and fire growth at the wrong time.

**Prerequisite logging:** before interpreting Phase 4.2, log per-layer EMA residual
traces in addition to the layer-aggregated trigger actually used for decisions.
Without per-layer telemetry, a multi-layer failure is not diagnosable.

| Run  | num_layers | growth / pruning config             | What it tests                                  |
|------|------------|-------------------------------------|------------------------------------------------|
| 4.2a | 2          | none                                | Two-layer phase-aware static baseline at 1000 steps |
| 4.2b | 2          | Phase 3.2f mode growth only         | Does the best single-layer mode-growth recipe transfer to depth? |

**Reporting:** final5 val_bpc, full event log, final `state_shape`, layer-aggregated EMA
trace used by the trigger, and the per-layer EMA residual traces. Specifically check:
- whether different layers plateau at materially different residual levels
- whether growth still fires after the initial transient rather than immediately
- whether event timing looks sensible relative to the deepest layer, shallowest layer,
  and the aggregated trigger

**Decision rule:** if run 4.2b beats run 4.2a by > 0.02 bpc and the event timing looks
structurally sensible, lift the `num_layers=1` restriction for future growth work. If
it loses or the trigger becomes hard to interpret across layers, keep all dynamic-growth
claims scoped to single-layer models and treat layer aggregation as an explicit follow-up
design question.

---

## Phase 4.1: Projector and FFN scalar hyperparameters

**8 runs** (4 configs × 2 seeds). Steps = 500. No growth. Uses the best static winner
as the base.

Purpose: test scalar hyperparameters that materially affect geometry and capacity but do
not fit naturally into the earlier TokenLift axis sweep.

**Prerequisite:** expose `phase_scale` as a training/CLI parameter before running this
phase. It exists in `TensorSignalProjector` but is not yet plumbed through the training
surface.

The base run already tests `phase_scale=π` and `ffn_expansion_factor=2`. This phase
adds alternative settings around those defaults:

| Run   | Variation                  | What it tests                                               |
|-------|----------------------------|-------------------------------------------------------------|
| 4.1a  | `phase_scale = π / 2`      | Reduced phase excursion; more local torque regime           |
| 4.1b  | `phase_scale = 2π`         | Full-circle coverage with stronger wraparound               |
| 4.1c  | `ffn_expansion_factor = 1` | Minimal FFN capacity; checks whether expansion is necessary |
| 4.1d  | `ffn_expansion_factor = 4` | Higher FFN capacity; standard transformer-style widening    |

**Decision rule:** adopt a non-default value only if it improves over the best static
winner by > 0.02 bpc. For `phase_scale`, also inspect stability and gradient pathologies
in logs, since larger phase ranges may improve expressivity but destabilize optimization.

---

## Phase 5: Axis isolation — per-mode norm

Full replication of Phase 1 under per-mode state normalization. Frobenius norm treats
the tensor state as a flat vector on a single hypersphere; per-mode normalization
independently scales each mode fiber (product-of-spheres manifold). These impose
different optimization landscapes and the TokenLift rankings from Phase 1 may not
transfer — particularly for phase-type variants (locked_wave, local_wave) whose phase
structure must survive state normalization across steps.

**Prerequisite:** expose `state_norm` as a CLI parameter with choices `frobenius`
(default, existing behavior) and `per_mode` before running this phase.

**18 runs** (6 configs × 3 seeds). Steps = 500. No growth. All hyperparameters
identical to Phase 1 except `state_norm = per_mode`.

| #  | token_magnitude_type      | phase_type  | token_phase    | What it tests                        |
|----|---------------------------|-------------|----------------|--------------------------------------|
| 19 | learned                   | rope        | none           | Per-mode baseline                    |
| 20 | inverse_frequency_learned | rope        | none           | Magnitude axis under per-mode norm   |
| 21 | learned                   | locked_wave | none           | Full phase coupling under per-mode   |
| 22 | learned                   | local_wave  | none           | Local phase coupling under per-mode  |
| 23 | learned                   | rope        | semantic       | Semantic token phase under per-mode  |
| 24 | learned                   | rope        | virtual_offset | Virtual offset under per-mode        |

**Decision rule:** same 0.02 bpc threshold as Phase 1, applied against run 19.

---

## Phase 6: Combine Phase 5 winners

Full replication of Phase 2 under per-mode norm. Same structure and decision rules as
Phase 2, using the Phase 5 axis winners as inputs.

**Up to 9 runs** (up to 3 configs × 3 seeds). Steps = 500. No growth.

| #  | token_magnitude_type       | phase_type             | token_phase                  | Rationale             |
|----|----------------------------|------------------------|------------------------------|-----------------------|
| 25 | [Phase 5 mag winner]       | [Phase 5 phase winner] | none                         | Best mag + best phase |
| 26 | [Phase 5 mag winner]       | rope                   | [Phase 5 token_phase winner] | Best mag + token phase |
| 27 | [Phase 5 mag winner]       | [Phase 5 phase winner] | [Phase 5 token_phase winner] | Full stack            |

Skip duplicates where axes collapsed to baseline.

**Decision rule for Phase 3/4 re-evaluation:** compare the Phase 6 winner against the
Phase 2 sequential winner (Frobenius). If the Phase 6 winner improves by > 0.02 bpc, adopt
per-mode norm as the new default and re-run Phases 3–4 under it. If the delta is
< 0.02, norm type is a minor confound; note it and proceed with Frobenius results.

---

## Phase 7: Optimization protocol validation

**15 runs** (5 configs × 3 seeds). Steps = 500. No growth. Uses the Phase 1 baseline
config (run 1: `learned` / `rope` / `none`).

Phases 1–2 assumed `lr=1e-3` with a fixed optimizer protocol. The pilot showed
val_bpc still materially improving at step 500, which is exactly where warmup + decay
may matter. This phase validates the training method, not just the scalar lr. It also
tests the two missing stability controls called out during review: gradient clipping
and weight decay.

| #   | lr    | warmup | decay    | min_lr_scale | grad_clip_norm | weight_decay | What it tests |
|-----|-------|--------|----------|--------------|----------------|--------------|---------------|
| 7.1 | 1e-3  | 0      | constant | 0.1          | None           | 0.0          | Legacy fixed-lr baseline |
| 7.2 | 1e-3  | 50     | cosine   | 0.1          | None           | 0.0          | Schedule only |
| 7.3 | 1e-3  | 50     | cosine   | 0.1          | 1.0            | 0.0          | Schedule + clipping |
| 7.4 | 1e-3  | 50     | cosine   | 0.1          | None           | 0.01         | Schedule + decay |
| 7.5 | 1e-3  | 50     | cosine   | 0.1          | 1.0            | 0.01         | Full proposed optimization protocol |

**Decision rule:** compare 7.2 against 7.1 first. If warmup + cosine improves by
> 0.02 bpc, adopt the scheduled optimizer as the new default. Then compare 7.3–7.5
against 7.2 to decide whether clipping and/or weight decay earn their keep.

**Secondary checks:** instability counts matter here. Record whether any run shows NaN,
Inf, sudden loss spikes, or strong seed sensitivity. If 7.5 matches the best bpc within
0.02 while being the most stable, prefer it as the default future protocol.

**Note:** this phase does not retroactively invalidate Phases 1–2 since all earlier
runs used the same legacy protocol. It does determine the optimizer protocol for all
new phases and any reruns of promoted checkpoints.

---

## Phase 8: State shape exploration

**Current run design:** **20 runs** (5 non-duplicate state shapes × 2 couplings × 2
seeds). Steps = 500. No growth. Uses the Phase 4C static quality recipe:

- `readout_type=phase_aware`
- `num_layers=2`
- `token_magnitude_type=inverse_frequency_learned`
- `phase_type=rope`
- `token_phase=semantic`
- `normalization_type=frobenius`

Run each shape under both `coupling_type=sequential` and `coupling_type=fft`. The FFT
arm is included as a backend stress check for shape geometry; prior spectral branches
remain closed unless FFT clears the same-shape sequential result by > 0.02 bpc.

Tests two independent questions about the tensor state structure.

**Sub-experiment A — Mode ordering** (`state_shape` permutations, rank 3, 24 elements):

Mode coupling in the mixer is sequential — information flows across modes in order.
The shape `(2,3,4)` (small→large) may favor different dynamics than `(4,3,2)` or a
balanced arrangement.

| Run  | state_shape | What it tests                              |
|------|-------------|--------------------------------------------|
| 8.A1 | (2,3,4)     | Small→large — current default              |
| 8.A2 | (4,3,2)     | Large→small — reversed coupling direction  |
| 8.A3 | (2,4,3)     | Large mode in the middle                   |

**Sub-experiment B — Rank vs mode size** (fixed ~24 state elements, varying rank):

Isolates whether expressivity comes from tensor rank (number of modes) or total state
capacity (product of mode sizes). All three shapes have the same or similar element
count.

| Run  | state_shape | Rank | Elements | What it tests                        |
|------|-------------|------|----------|--------------------------------------|
| 8.B1 | (2,3,4)     | 3    | 24       | Current — rank-3 reference; covered by 8.A1 |
| 8.B2 | (4,6)       | 2    | 24       | Same capacity, one fewer mode        |
| 8.B3 | (24,)       | 1    | 24       | Degenerate vector state — rank-1 floor |

Run 8.B3 is especially informative: if a flat vector state at the same parameter count
matches or beats the rank-3 tensor, the multi-mode structure is not contributing.

For the current run, skip the duplicate 8.B1 execution because 8.A1 is the same
`state_shape=(2,3,4)`.

---

## Phase 9: Sequence length

**6 runs** (3 configs × 2 seeds). Steps = 500. No growth. Uses the best static winner:
the Phase 2U winner if spectral coupling helped, otherwise the Phase 2 sequential
winner.

The model is stateful and accumulates context across the sequence rather than attending
over a fixed window. It may benefit from longer sequences more than a window-based model
would. Tests whether the RNN-like state dynamics can exploit longer context.

| #   | seq_len | What it tests                                    |
|-----|---------|--------------------------------------------------|
| 9.1 | 64      | Current default — replication                    |
| 9.2 | 128     | 2× context — does the state extract more signal? |
| 9.3 | 256     | 4× context — diminishing returns or still scaling? |

**Note:** eval_batches may need reducing at seq_len=256 to keep eval time reasonable
(each batch is 4× larger). Halving to 8 eval batches keeps token count similar.

---

## Phase 9.1: Streaming inference benchmark

**Post-hoc evaluation only. No new training runs.**

Run this benchmark on every promoted checkpoint:
- Phase 2 sequential winner
- tuned spectral winner, if one exists
- Phase 3.5 full-system winner, if dynamic growth remains in scope

Use `benchmark_prompt_lengths=(32,128,512,1024)` and `benchmark_new_tokens=256`.
Report:
- prompt throughput (`prompt_tokens_per_second`)
- decode throughput (`decode_tokens_per_second`)
- peak memory (`peak_memory_bytes`)

**Decision rule:** the O(1) decode claim is supported only if decode throughput and
peak memory stay approximately flat as prompt length increases. If decode speed or
memory grows materially with prompt length, do not repeat the O(1) claim in writeups.

---

## Phase 9.2: Generation quality evaluation

**Post-hoc evaluation only. No new training runs.**

Run this on every promoted checkpoint with:
- `generation_eval_samples=4`
- `generation_prompt_len=64`
- `generation_new_tokens=256`
- `generation_temperature=0.8`
- `generation_top_k=20`

For each checkpoint, save:
- four prompt/continuation pairs
- `distinct-1`
- `distinct-2`
- a short manual note on local coherence, repetition, and stylistic plausibility

**Decision rule:** a perplexity win is not promoted if samples collapse into
obvious repetition, unreadable character soup, or low-diversity continuations across
all four prompts. Generation quality is a gate, not just a nice-to-have appendix.

---

## Phase 10: Hidden size scaling

Full replication of Phase 1 at `hidden_size=64`. All other hyperparameters identical
to Phase 1.

**18 runs** (6 configs × 3 seeds). Steps = 500. No growth.

Phase 1 rankings at `hidden_size=32` (16 complex dimensions) may not hold at 64. Phase
effects in particular — locked_wave, local_wave, semantic token phase — require the
embedding space to be rich enough to encode meaningful phase variation. If rankings
shift at 64, the Phase 1 winner is architecture-sensitive and the 32-dim results should
not be generalized.

| #    | token_magnitude_type      | phase_type  | token_phase    |
|------|---------------------------|-------------|----------------|
| 10.1 | learned                   | rope        | none           |
| 10.2 | inverse_frequency_learned | rope        | none           |
| 10.3 | learned                   | locked_wave | none           |
| 10.4 | learned                   | local_wave  | none           |
| 10.5 | learned                   | rope        | semantic       |
| 10.6 | learned                   | rope        | virtual_offset |

**Decision rule:** compare Phase 10 rankings against Phase 1 rankings. If the winner
changes, run a Phase 10.2 combination step (mirrors Phase 2) using Phase 10 axis
winners. If rankings are stable, Phase 1 results generalize across this hidden size
range.

---

## Run summary

| Phase | Purpose                              | Configs | Seeds | Runs  | Status     |
|-------|--------------------------------------|---------|-------|-------|------------|
| 0     | Pilot / sanity check                 | 1       | 1     | 1     | ✓ Complete |
| 1     | Axis isolation (Frobenius)           | 6       | 3     | 18    | ✓ Complete |
| 2     | Combination (Frobenius)              | ≤3      | 3     | ≤9    | ✓ Complete |
| 2S    | Spectral coupling screen             | 5       | 3     | 15    |            |
| 2S.1  | Wavelet depth calibration            | 3       | 3     | 9     |            |
| 2S.2  | Spectral filter calibration          | 3       | 3     | 9     |            |
| 2S.3  | Spectral screen at higher state dim  | 5       | 3     | 15    |            |
| 2S.4  | FFT learned/anisotropic gain ablation | 5      | 3     | 15    |            |
| 2D    | Dynamic rank/mode backend adaptation | 3       | 3     | 9     |            |
| 2T    | Axis isolation under tuned spectral backend | 6 | 3 | 18 |            |
| 2U    | Combination under tuned spectral backend | ≤3 | 3 | ≤9 |            |
| 3     | Static baseline at 1000 steps        | 1       | 2     | 2     |            |
| 3.1   | Residual EMA calibration             | 3       | 2     | 6     |            |
| 3.2   | Mode growth — residual EMA trigger   | 5       | 2     | 10    |            |
| 3.3   | Rank growth — residual EMA trigger   | 3       | 2     | 6     |            |
| 3.4   | Pruning                              | 3       | 2     | 6     |            |
| 3.5   | Full system (grow + prune)           | 2       | 2     | 4     |            |
| 4     | Ablations                            | ≤6      | 2     | ≤12   |            |
| 4.2   | Multi-layer growth sanity check      | 2       | 2     | 4     |            |
| 4.1   | Projector / FFN hyperparams          | 4       | 2     | 8     |            |
| 5     | Axis isolation (per-mode norm)       | 6       | 3     | 18    |            |
| 6     | Combination (per-mode norm)          | ≤3      | 3     | ≤9    |            |
| 7     | Optimization protocol validation     | 5       | 3     | 15    |            |
| 8     | State shape exploration              | 6       | 2     | 12    |            |
| 9     | Sequence length                      | 3       | 2     | 6     |            |
| 9.1   | Streaming benchmark (post-hoc)       | promoted checkpoints | n/a | 0 |     |
| 9.2   | Generation quality eval (post-hoc)   | promoted checkpoints | n/a | 0 |     |
| 10    | Hidden size scaling                  | 6       | 3     | 18    |            |
| **Total** |                                  |         |       | **≤251** |         |

---

## Phase 11: Hybrid architecture — sanity scale

**Goal:** Does the hybrid (Reciprocator + local attention) learn faster and to lower bpc than the pure Reciprocator at the same parameter count? Is the two-timescale training regime (stateful + gradient) working?

**Corpus:** `greek_classics` at `max_chars=100,000` for continuity with prior phases.

**Config:**

| Parameter | Value |
|---|---|
| num_layers | 6 (Reciprocator) |
| attention_every_k | 3 |
| attention_num_heads | 4 |
| attention_window | 128 |
| hidden_size | 256 |
| state_shape | (4, 4, 4) |
| normalization_type | per_mode |
| enable_self_relation | True |
| token_magnitude_type | inverse_frequency_learned |
| phase_type | rope |
| readout_type | phase_aware |
| stateful_training | True |
| seq_len | 128 |
| batch_size | 8 |
| steps | 2000 |
| eval_every | 100 |
| learning_rate | 1e-3 |
| lr_warmup_steps | 100 |
| lr_decay_style | cosine |
| min_lr_scale | 0.1 |
| grad_clip_norm | 1.0 |
| weight_decay | 0.01 |
| seeds | [0, 1, 2] |

**Runs:**

| Run | Description | attention_position | Attn blocks |
|---|---|---|---|
| 11A | Hybrid as above | `after` | 1 |
| 11B | Pure Reciprocator, nearest current parameter match (`num_layers=7`) | n/a | 0 |
| 11C | Hybrid with `stateful_training=False` | `after` | 1 |
| 11D | Hybrid with attention first in each group | `before` | 2 |

**Attention ordering (11A vs 11D):**

- `"after"` (11A): `[R,R,R,A, R,R,R]` — attention is inserted after each completed k-group except the last one, so the next Reciprocator group processes the attention-enhanced state.
- `"before"` (11D): `[A,R,R,R, A,R,R,R]` — attention is inserted before each k-group, so Reciprocators within the group process the attention-resolved state and fold it into long-range memory.

`"before"` has one more attention block than `"after"` for the same `num_layers`. Report exact non-embedding parameter counts for both; the difference should be small (one attention block ≈ `4 × hidden_size²` additional parameters).

Parameter matching for 11B: count non-embedding parameters in 11A and set `num_layers` for the pure Reciprocator to the nearest match. Under the current model code, 11A has `6,369,761` non-embedding parameters and the nearest pure match is `num_layers=7` with `6,797,778`. Recompute if the block definitions change. Report exact param counts for all runs.

**Primary metric:** mean `val_bpc` over the final 5 eval checkpoints.

**Decision rules:**
- If 11A beats 11B by > 0.02 bpc: hybrid architecture is contributing; proceed to Phase 12.
- If 11A beats 11C by > 0.02 bpc: stateful training is contributing; keep it as default.
- If 11D beats 11A by > 0.02 bpc: adopt `attention_position="before"` as the default for Phase 12+. The interpretation: attention-first means each Reciprocator group folds already-resolved local context into long-range state, which is more efficient for coding/math where variable bindings should be settled before state accumulation.
- If 11A and 11D are within 0.02 bpc: ordering does not matter at this scale; keep `"after"` as default (fewer attention blocks, fewer parameters).
- If neither 11A nor 11D beats 11B: attention is not helping; diagnose before scaling.

**Secondary checks:** streaming benchmark on all three runs at `benchmark_prompt_lengths=(128, 512, 1024, 2048)`. Confirm O(1) decode holds for 11A and 11B; confirm 11A KV cache does not grow unboundedly (window clipping is working).

---

## Phase 12: Small production scale — coding/math quality

**Goal:** Does the hybrid architecture achieve meaningful coding/math language modeling quality? Is the per-token loss on code/math lower than a matched Transformer or Mamba baseline?

**Prerequisite:** add a coding corpus. Recommended: a Python subset (100K–500K chars) from a public domain source (e.g., Python standard library source, public educational Python code). Math corpus: ProofWiki or a filtered ArXiv abstract subset. Both should be added to the `corpora/` directory using the existing `build_text_dataset` pipeline. At minimum, add one coding corpus before running this phase.

**Config:**

| Parameter | Value |
|---|---|
| num_layers | 12 (Reciprocator) |
| attention_every_k | 3 → 3 LocalAttentionBlock insertions with default `attention_position="after"`; 4 if Phase 11 promotes `"before"` |
| attention_num_heads | 8 |
| attention_window | 256 |
| hidden_size | 512 |
| state_shape | (8, 8, 4) |
| normalization_type | per_mode |
| enable_self_relation | True |
| token_magnitude_type | inverse_frequency_learned |
| phase_type | rope |
| readout_type | phase_aware |
| stateful_training | True |
| seq_len | 256 |
| batch_size | 8 |
| steps | 10,000 |
| eval_every | 500 |
| learning_rate | 1e-3 |
| lr_warmup_steps | 500 |
| lr_decay_style | cosine |
| min_lr_scale | 0.1 |
| grad_clip_norm | 1.0 |
| weight_decay | 0.01 |
| seeds | [0, 1] |

**Corpus:** coding corpus (see prerequisite). Also run on `greek_classics` for cross-corpus comparison.

**Reporting metric:** mean `val_bpc` over the final 5 eval checkpoints. Report separately for the coding corpus and `greek_classics`.

**Decision rule:** if coding-corpus bpc is competitive with the Transformer baseline (Phase B, run B2) within 0.1 bpc, proceed to Phase 13. If not, diagnose: check whether the attention window is too small, whether stateful training is actually helping on code structure, and whether the state_shape is appropriate for the coding task.

---

## Phase 13: Medium scale — scaling check

**Goal:** Does quality improve predictably when going from 12 to 18 Reciprocator layers at the same hidden_size? Is the architecture compute-efficient (loss-per-FLOP competitive)?

**Config:** identical to Phase 12 except `num_layers=18` (Reciprocator), `attention_every_k=3` → 5 LocalAttentionBlock insertions with default `attention_position="after"`; 6 if Phase 11 promotes `"before"`.

**Runs:**

| Run | num_layers (Recip) | Attn blocks | Steps | Goal |
|---|---|---|---|---|
| 13A | 18 | 5 (`after`) / 6 (`before`) | 10,000 | Scaling check |
| 13B | 12 (Phase 12 winner) | 3 (`after`) / 4 (`before`) | 10,000 | Comparison baseline |

**Metrics:** val_bpc and training FLOPs (estimate: `6 × params × tokens_seen`). Plot bpc vs. FLOPs for both runs to check scaling efficiency.

**Decision rule:** if 13A beats 13B by > 0.05 bpc (larger margin required at this cost level), the architecture scales positively. If the improvement is < 0.05 bpc per 50% more depth, the architecture is depth-saturated at this hidden_size — increasing hidden_size or state_shape is the next lever.

---

## Phase B: Baseline comparisons

**Goal:** At matched parameter count and training compute, how does the hybrid Reciprocator compare to a standard Transformer and a Mamba-style SSM?

### B.1 — Baseline implementations

Three architectures are compared. Each must be implemented to a fair standard:

**Transformer baseline** (implement as a reference model in `src/reciprocator/baselines.py`):
- Pre-norm causal transformer with RoPE positional encoding
- Multi-head attention with causal mask (no window restriction — full causal)
- SwiGLU FFN (`ffn_expansion_factor=8/3` to match standard)
- No dropout
- Same `hidden_size`, `num_layers` tuned to match non-embedding parameter count
- Training: random batch sampling (no stateful training — Transformer has no recurrent state), same optimizer protocol as Phase 7 winner

**Mamba baseline** (implement or reference):
- Selective state space model (Mamba-1 architecture: input-dependent SSM with selective scan)
- Same `hidden_size`, depth tuned to parameter match
- State dimension `d_state=16` (standard Mamba default)
- Training: stateful (SSM state carried across chunks) OR standard (depends on reference implementation)
- Note: if implementing from scratch is too costly, use the simplified S4/DSS formulation. Document which variant was used.

**Hybrid Reciprocator** (Phase 12 winner config):
- As specified in Phase 12.

### B.2 — Parameter matching protocol

For each pair of comparisons, match non-embedding parameter counts within 5%. Report exact non-embedding param count for all three architectures. Compute budget matched by training for the same number of gradient steps on the same corpus at the same batch_size × seq_len.

Transformer non-embedding params per layer: `~12 × D²` (attention) + `~8 × D²` (SwiGLU FFN) ≈ `20 × D²`.

Reciprocator non-embedding params per block: varies by state_shape; estimate directly from `sum(p.numel() for p in model.parameters()) - embedding_params`.

For `hidden_size=256`, target approximately 5–10M non-embedding parameters for the sanity comparison (Phase B against Phase 11). For `hidden_size=512`, target 20–40M for the production comparison (Phase B against Phase 12).

### B.3 — Runs

**Sanity comparison (against Phase 11A, hidden_size=256, `greek_classics` corpus):**

| Run | Architecture | Params | seq_len | Steps |
|---|---|---|---|---|
| B1 | Transformer (param-matched) | ≈ Phase 11A | 128 | 2000 |
| B2 | Mamba (param-matched) | ≈ Phase 11A | 128 | 2000 |
| B3 | Hybrid Reciprocator (Phase 11A) | reference | 128 | 2000 |

**Production comparison (against Phase 12, hidden_size=512, coding corpus):**

| Run | Architecture | Params | seq_len | Steps |
|---|---|---|---|---|
| B4 | Transformer (param-matched) | ≈ Phase 12 | 256 | 10,000 |
| B5 | Mamba (param-matched) | ≈ Phase 12 | 256 | 10,000 |
| B6 | Hybrid Reciprocator (Phase 12) | reference | 256 | 10,000 |

Seeds: [0, 1] for all. Report mean ± std across seeds.

### B.4 — Metrics

**Quality:**
- Mean `val_bpc` over the final 5 eval checkpoints.
- Learning curve: bpc at steps {500, 1000, 2000, 5000, 10000} — convergence speed matters, not just final quality.

**Streaming inference (critical for the O(1) claim):**

| Metric | Transformer | Mamba | Hybrid Reciprocator |
|---|---|---|---|
| Decode speed (tok/s) at prompt=128 | — | — | — |
| Decode speed (tok/s) at prompt=1024 | — | — | — |
| Decode speed (tok/s) at prompt=4096 | — | — | — |
| Peak memory at prompt=128 | — | — | — |
| Peak memory at prompt=4096 | — | — | — |

The Transformer's decode speed and memory should grow with prompt length (O(T) KV cache). Mamba and Reciprocator should not. If the Reciprocator's decode speed or memory grows with prompt length, the O(1) claim is not supported and the KV cache windowing in `LocalAttentionBlock` needs investigation.

**Coding-specific quality:**
For coding corpus runs, additionally report:
- Distinct-1 and distinct-2 on generated code continuations (4 samples, 256 new tokens each).
- Whether generated code is syntactically plausible (manual spot-check, not automated).

### B.5 — Decision rule

The Reciprocator is the preferred architecture if:
1. `val_bpc` within 0.1 of the Transformer on the coding corpus (quality parity), AND
2. Decode speed is materially faster than Transformer at prompt length ≥ 512 (efficiency advantage), AND
3. Peak memory at prompt=4096 is materially lower than Transformer.

If the Reciprocator loses quality by > 0.1 bpc against the Transformer, document the gap clearly. Do not present the architecture as production-ready on coding/math until Phase 12 and Phase B both show quality competitive within 0.1 bpc.

Mamba is the key competitor for efficiency claims. If Mamba matches the Reciprocator's bpc while being simpler, the Reciprocator's relational/rotational inductive bias is not earning its complexity. Report the Mamba comparison without spin.

---

## Run summary (updated)

| Phase | Purpose                              | Configs | Seeds | Runs  | Status     |
|-------|--------------------------------------|---------|-------|-------|------------|
| 0     | Pilot / sanity check                 | 1       | 1     | 1     | ✓ Complete |
| 1     | Axis isolation (Frobenius)           | 6       | 3     | 18    | ✓ Complete |
| 2     | Combination (Frobenius)              | ≤3      | 3     | ≤9    | ✓ Complete |
| 2S    | Spectral coupling screen             | 5       | 3     | 15    |            |
| 2S.1  | Wavelet depth calibration            | 3       | 3     | 9     |            |
| 2S.2  | Spectral filter calibration          | 3       | 3     | 9     |            |
| 2S.3  | Spectral screen at higher state dim  | 5       | 3     | 15    |            |
| 2S.4  | FFT learned/anisotropic gain ablation | 5      | 3     | 15    |            |
| 2D    | Dynamic rank/mode backend adaptation | 3       | 3     | 9     |            |
| 2T    | Axis isolation under tuned spectral backend | 6 | 3 | 18 |            |
| 2U    | Combination under tuned spectral backend | ≤3 | 3 | ≤9 |            |
| 3     | Static baseline at 1000 steps        | 1       | 2     | 2     |            |
| 3.1   | Residual EMA calibration             | 3       | 2     | 6     |            |
| 3.2   | Mode growth — residual EMA trigger   | 5       | 2     | 10    |            |
| 3.3   | Rank growth — residual EMA trigger   | 3       | 2     | 6     |            |
| 3.4   | Pruning                              | 3       | 2     | 6     |            |
| 3.5   | Full system (grow + prune)           | 2       | 2     | 4     |            |
| 4     | Ablations                            | ≤6      | 2     | ≤12   |            |
| 4.2   | Multi-layer growth sanity check      | 2       | 2     | 4     |            |
| 4.1   | Projector / FFN hyperparams          | 4       | 2     | 8     |            |
| 5     | Axis isolation (per-mode norm)       | 6       | 3     | 18    |            |
| 6     | Combination (per-mode norm)          | ≤3      | 3     | ≤9    |            |
| 7     | Optimization protocol validation     | 5       | 3     | 15    |            |
| 8     | State shape exploration              | 6       | 2     | 12    |            |
| 9     | Sequence length                      | 3       | 2     | 6     |            |
| 9.1   | Streaming benchmark (post-hoc)       | promoted | n/a  | 0     |            |
| 9.2   | Generation quality eval (post-hoc)   | promoted | n/a  | 0     |            |
| 10    | Hidden size scaling                  | 6       | 3     | 18    |            |
| 11    | Hybrid sanity scale (hidden=256)     | 4       | 3     | 12    |            |
| 12    | Small production scale (hidden=512)  | 1       | 2     | 2     |            |
| 13    | Medium scale check (18 layers)       | 2       | 2     | 4     |            |
| B     | Baseline comparisons (Transformer, Mamba) | 6  | 2     | 12    |            |
| **Total** |                                  |         |       | **≤272** |         |

---

## Analysis notes

- Always report mean ± std across seeds, not just the mean. A "winner" with high
  variance may not be meaningfully better than second place.
- Treat the optimizer protocol as part of the experimental condition, not invisible
  plumbing. Report warmup, decay, clipping, and weight decay alongside every result.
- For Phase 3, also report the growth event log (when growth fired, what shape
  resulted) together with the EMA-smoothed mode residual traces, to understand whether
  the residual threshold and the rank-growth loss backstop are well-calibrated.
- Coupling type is not assumed independent of TokenLift axes. Phase 2S answers "does
  the spectral family help at all?"; Phases 2S.1 and 2S.2 then calibrate depth and the
  four-parameter spectral filter envelope on the strongest spectral candidates; Phases
  2T and 2U rerun the TokenLift search under the tuned spectral backend before
  declaring the best static winner.
- Dynamic growth and pruning support both sequential and spectral coupling backends.
  Phases 3–3.5 remain on the Phase 2 sequential winner for baseline continuity; Phase
  2D is the cross-backend dynamic adaptation check.
- Dynamic growth conclusions are currently scoped to `num_layers=1`. If static depth
  helps, Phase 4.2 is the gate for deciding whether layer-aggregated residual triggers
  also work with depth.
- The streaming benchmark and generation-quality pass are promotion gates for
  checkpoints, not optional afterthoughts. If a checkpoint wins on bpc but fails either
  gate, document that explicitly and do not present it as an unqualified best model.
- `phase_scale` is not currently surfaced through the training configuration. The plan
  now treats that as an explicit prerequisite for Phase 4.1 rather than silently
  assuming the sweep can be run.
- If Phase 1 finds no axis beats baseline by 0.02 bpc, Phase 2 collapses to a single
  run (identical to baseline) and can be skipped. This is a valid and informative result.
- Phases 5–6 mirror Phases 1–2 exactly under per-mode norm. The comparison between
  Phase 2 and Phase 6 winners is the key norm decision. If per-mode wins by > 0.02 bpc,
  re-run Phases 3–4 under per-mode. If not, Frobenius stands and Phases 5–6 are
  informative but not action-forcing.

## Downstream: RL mathematical reasoning

A separate experimental track — [`docs/rl/test-plan.md`](./rl/test-plan.md) — uses the
architecture produced by this plan as the backbone for RL training on a custom Lisp
dialect. That plan teaches mathematical reasoning (arithmetic → symbolic algebra → proof
construction) using GRPO-style policy gradient updates with a staged curriculum.

The RL plan consumes the best architecture from this plan's phases. If later phases
(11-13) produce a significantly different architecture, the RL plan's base configuration
should be updated to match. The two plans share `ReciprocatorLM` and `training.py` but
use different training objectives, data sources, and evaluation metrics.
