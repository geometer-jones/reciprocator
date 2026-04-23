# Training Guide

This guide documents the practical training workflow for the current
`reciprocator` codebase.

## What the training script does

The main entry point is:

```bash
python3 scripts/train.py --help
```

At a high level the script:

1. loads a bundled corpus
2. builds a character-level tokenizer and train/validation split
3. constructs a `ReciprocatorLM`
4. trains with the settings in `TrainingConfig`
5. reports train/validation metrics during the run
6. optionally writes checkpoints, residual diagnostics, generation samples, and
   runtime benchmarks

The implementation lives in:

- `scripts/train.py`
- `src/reciprocator/training.py`

## Basic run

Small smoke run:

```bash
python3 scripts/train.py \
  --corpus plato_jowett \
  --device auto \
  --steps 50 \
  --eval-every 10 \
  --batch-size 8 \
  --seq-len 64 \
  --hidden-size 32 \
  --state-shape 2,3,4
```

Checkpointed run with an explicit name:

```bash
python3 scripts/train.py \
  --corpus plato_jowett \
  --device auto \
  --steps 500 \
  --eval-every 50 \
  --batch-size 8 \
  --seq-len 64 \
  --hidden-size 32 \
  --state-shape 2,3,4 \
  --run-name phase1_baseline \
  --checkpoint-every 100
```

With `--run-name`, the script writes outputs under `runs/<run-name>/` by default.

## Important flag groups

### Data and optimization

- `--corpus`
- `--max-chars`
- `--val-fraction`
- `--batch-size`
- `--seq-len`
- `--steps`
- `--eval-every`
- `--eval-batches`
- `--lr`
- `--lr-warmup-steps`
- `--lr-decay-style`
- `--min-lr-scale`
- `--grad-clip-norm`
- `--weight-decay`
- `--seed`

### Base model shape

- `--hidden-size`
- `--state-shape`
- `--num-layers`
- `--ffn-expansion-factor`
- `--readout-type`
- `--normalization-type`

### TokenLift controls

- `--token-magnitude-type`
- `--phase-type`
- `--token-phase`

These control how token ids are lifted into complex hidden vectors before they
enter the reciprocator blocks.

### Mixer and coupling controls

- `--coupling-type`
- `--enable-self-relation`
- `--low-frequency-gain`
- `--low-frequency-sigma`
- `--high-frequency-gain`
- `--high-frequency-cutoff`
- `--dynamic-spectral-gains`
- `--anisotropic-spectral-gains`
- `--wavelet-levels`
- `--chunk-size`
- `--track-chunk-drift`

Supported coupling backends today:

- `sequential`
- `fft`
- `dwt`
- `wavelet_packet`
- `wavelet_packet_max_gauge`

`--dynamic-spectral-gains` applies only to spectral backends (`fft`, `dwt`,
`wavelet_packet`, and `wavelet_packet_max_gauge`). It keeps the fixed spectral
filter as the base envelope, then adds a zero-initialized low-rank projector
conditioned on the current complex coupling signal. The projector width is
`gain_projector_rank` in `TrainingConfig`.

For FFT coupling, `--anisotropic-spectral-gains` makes that dynamic projector a
full coordinatewise frequency-grid modulation instead of the default radial
sampled modulation.

### Extra model paths

- `--enable-anticipator-relation`
- `--enable-cross-layer-state`

These flags expose the optional non-baseline paths currently wired through the
CLI into `ReciprocatorLM`.

Local attention support also exists in the Python model/config layer, but the
current `scripts/train.py` parser does not expose attention-specific flags.
If you want to experiment with attention today, do it programmatically through
`TrainingConfig` and `train_model()` rather than through the CLI.

### Dynamic growth and pruning

- `--dynamic-mode-growth`
- `--dynamic-rank-growth`
- `--dynamic-mode-pruning`
- `--dynamic-rank-pruning`
- `--max-state-shape`
- `--max-rank`
- `--growth-check-interval`
- `--growth-residual-threshold`
- `--residual-saturate-threshold`
- `--growth-residual-ema-decay`
- `--min-checks-before-first-growth`
- `--rank-growth-loss-ceiling`
- `--prune-threshold`
- `--prune-sustain-steps`
- `--prune-min-steps`
- `--mode-init`
- `--rank-init`
- `--record-residual-diagnostics`
- `--diagnostics-out`

These are research features rather than stable production controls. When used
with `--record-residual-diagnostics`, the run can emit a JSON diagnostics file
describing residual and redundancy signals across growth checks.

## Generation and benchmark outputs

You can attach generation and runtime evaluation to a training run:

```bash
python3 scripts/train.py \
  --corpus plato_jowett \
  --steps 100 \
  --eval-every 20 \
  --generation-eval-samples 4 \
  --generation-prompt-len 64 \
  --generation-new-tokens 128 \
  --generation-temperature 0.8 \
  --generation-top-k 20 \
  --benchmark-prompt-lengths 32,128,512 \
  --benchmark-new-tokens 128
```

This causes the training result to include:

- sampled continuations with `distinct_1` and `distinct_2`
- prompt/decode throughput measurements
- a best-effort peak memory metric

## Run outputs

Depending on flags, a run may produce:

- `runs/<run-name>/config.json`
- `runs/<run-name>/checkpoint_step_XXXXXX.pt`
- `runs/<run-name>/residual_diagnostics.json`
- summary files or external logs created by experiment scripts

`--checkpoint-out` can also write a final checkpoint to an explicit path outside
the run directory.

## Common workflows

### 1. Baseline training run

```bash
python3 scripts/train.py \
  --corpus plato_jowett \
  --steps 500 \
  --eval-every 50 \
  --hidden-size 32 \
  --state-shape 2,3,4 \
  --run-name baseline
```

### 2. Compare coupling backends

```bash
python3 scripts/train.py \
  --corpus plato_jowett \
  --steps 500 \
  --eval-every 50 \
  --state-shape 2,3,4 \
  --coupling-type fft \
  --run-name fft_screen
```

Swap `fft` for `sequential`, `dwt`, `wavelet_packet`, or
`wavelet_packet_max_gauge`.

To compare adaptive spectral filters, keep the same backend and add
`--dynamic-spectral-gains`:

```bash
python3 scripts/train.py \
  --corpus plato_jowett \
  --steps 500 \
  --eval-every 50 \
  --state-shape 2,3,4 \
  --coupling-type fft \
  --dynamic-spectral-gains \
  --anisotropic-spectral-gains \
  --run-name fft_dynamic_spectral_screen
```

### 3. Growth/pruning experiment

```bash
python3 scripts/train.py \
  --corpus plato_jowett \
  --steps 1000 \
  --eval-every 50 \
  --state-shape 2,3,4 \
  --dynamic-mode-growth \
  --max-state-shape 4,4,5 \
  --growth-check-interval 50 \
  --growth-residual-threshold 0.12 \
  --growth-residual-ema-decay 0.8 \
  --record-residual-diagnostics \
  --run-name growth_probe
```

## Programmatic API

The same workflow is available directly from Python through
`src/reciprocator/training.py`.

Important entry points:

- `TrainingConfig`
- `build_text_dataset()`
- `build_corpus_dataset()`
- `sample_causal_lm_batch()`
- `train_model()`

Minimal example:

```python
from reciprocator.training import TrainingConfig, train_model

config = TrainingConfig(
    corpus_name="plato_jowett",
    steps=10,
    eval_every=5,
    hidden_size=32,
    state_shape=(2, 3, 4),
)

result = train_model(config)
print(result.train_metrics[-1])
```

## Reading results

The main scalar metrics surfaced by the training loop are:

- `loss`
- `accuracy`
- `perplexity`
- `bpc` (bits per character)

For experiment-tracking conventions already used in this repository, see:

- [`../test_plan.md`](../test_plan.md)
- [`../lab-book.md`](../lab-book.md)
