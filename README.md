# Reciprocator

Reciprocator is an experimental PyTorch language-modeling scaffold built around a
complex-valued recurrent tensor state. The repository bundles philosophy corpora,
a character-level training pipeline, several reciprocator coupling backends, and a
small test suite for validating model behavior.

The codebase is set up as a research sandbox rather than a polished training
framework. The focus is on trying architectural ideas quickly, recording runs under
`runs/`, and keeping the core modeling code small enough to inspect directly.

## What is in the repo

- A `src/` package containing the model, mixer, complex-valued layers, training
  loop, and corpus helpers.
- A CLI training entry point at `scripts/train.py`.
- Bundled training corpora under `corpora/`.
- Saved experimental artifacts under `runs/`.
- A `pytest` suite covering the package API and training script behavior.
- Research notes in `test_plan.md` and the corpus-specific READMEs.

## Repository layout

```text
reciprocator/
├── corpora/                  # Bundled training text + provenance metadata
├── runs/                     # Saved checkpoints, configs, logs, diagnostics
├── scripts/train.py          # CLI entry point for training runs
├── src/reciprocator/
│   ├── complex_ops.py        # Complex normalization and linear/activation layers
│   ├── corpora.py            # Bundled corpus registry and file access helpers
│   ├── mixer.py              # Reciprocator tensor-state mixer and coupling backends
│   ├── model.py              # Token lift, blocks, attention, readouts, LM wrapper
│   ├── training.py           # Dataset building, training loop, eval, generation
│   └── rl/
│       ├── lisp_eval.py      # Custom Lisp s-expression parser and evaluator
│       ├── problem_gen.py    # Problem generator with staged difficulty
│       ├── reward.py         # Reward function with partial credit
│       ├── curriculum.py     # Competency-driven curriculum controller
│       ├── phase_monitor.py  # Complex-valued state phase trajectory tracking
│       └── training.py       # GRPO training loop for mathematical reasoning
├── tests/                    # Pytest coverage for model, training, corpora, CLI
├── pyproject.toml            # setuptools package config and pytest settings
└── test_plan.md              # Experiment plan and run-tracking notes
```

## Architecture summary

The main data flow is:

1. Text is loaded from a bundled corpus or provided directly as a string.
2. `CharTokenizer` in `training.py` builds a character vocabulary and encodes the
   corpus into train/validation token streams.
3. `TokenLift` in `model.py` maps token ids into complex hidden vectors using
   configurable magnitude and phase schemes.
4. `ReciprocatorBlock` applies a `ReciprocatorMixer` over a complex tensor state,
   then a complex feed-forward layer. Optional local causal attention can be
   inserted between blocks.
5. `ReciprocatorLM` reads complex hidden states back to token logits through either
   a magnitude-only or phase-aware readout.
6. `train_model()` in `training.py` handles optimization, evaluation, text
   generation samples, runtime benchmarking, checkpoints, and optional residual
   diagnostics for dynamic growth/pruning experiments.

## Core concepts

### Complex-valued token lifting

`TokenLift` supports a few experimental axes:

- Token magnitude: `learned`, `inverse_frequency`, `inverse_frequency_learned`
- Positional phase: `rope`, `locked_wave`, `local_wave`
- Token phase: `none`, `semantic`, `virtual_offset`, `semantic_virtual_offset`

### Mixer backends

The reciprocator mixer supports multiple coupling modes:

- `sequential`
- `fft`
- `dwt`
- `wavelet_packet`
- `wavelet_packet_max_gauge`

It also supports chunked forward passes, optional self-relation terms, and
different complex-state normalization modes (`frobenius` and `per_mode`).
Spectral backends can opt into learned, signal-conditioned filter envelopes with
`dynamic_spectral_gains` / `--dynamic-spectral-gains`; the scalar spectral
parameters remain the fixed base envelope, and the learned path is inert at
initialization. FFT coupling can further opt into full coordinatewise dynamic
frequency-grid modulation with `anisotropic_spectral_gains` /
`--anisotropic-spectral-gains`; otherwise dynamic FFT gains remain radial.

### Training and experiments

The training stack is character-level and intentionally minimal:

- character tokenizer
- contiguous train/validation split
- causal LM batching
- train/validation loss, accuracy, perplexity, and bits-per-character
- optional text generation samples
- optional streaming runtime benchmarks
- optional checkpointing and residual diagnostics

## Requirements

- Python 3.9+
- PyTorch

The repository uses `setuptools` via `pyproject.toml` and exposes the package from
`src/`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

## Running tests

```bash
python3 -m pytest
```

The test suite lives in `tests/` and covers:

- complex normalization and layer behavior
- corpus registry and bundled file access
- mixer coupling backends and chunked execution
- `ReciprocatorLM` configuration paths
- training-loop smoke tests and CLI argument parsing

## Quick start

Run a small training job on a bundled corpus:

```bash
python3 scripts/train.py \
  --corpus greek_classics \
  --device auto \
  --steps 50 \
  --eval-every 10 \
  --batch-size 8 \
  --seq-len 64 \
  --hidden-size 32 \
  --state-shape 2,3,4 \
  --run-name smoke_run
```

That command will:

- load the bundled `greek_classics` corpus
- build a character vocabulary and train/validation split
- train a small `ReciprocatorLM`
- write run config and checkpoints under `runs/smoke_run/` when checkpointing is enabled

See the full CLI surface with:

```bash
python3 scripts/train.py --help
```

## Bundled corpora

Three corpora are registered today:

- `chinese_classics`
- `greek_classics`
- `lisp_math`

Each corpus directory contains:

- a combined training text file
- a `README.md` describing the source bundle
- a `sources.tsv` provenance table

Example:

```python
from reciprocator import available_corpora, read_corpus_text

print([corpus.name for corpus in available_corpora()])
text = read_corpus_text("lisp_math")
print(len(text))
```

## Programmatic usage

```python
from reciprocator.model import ReciprocatorLM
from reciprocator.training import TrainingConfig, train_model

model = ReciprocatorLM(
    vocab_size=128,
    hidden_size=32,
    state_shape=(2, 3, 4),
    num_layers=1,
)

config = TrainingConfig(
    corpus_name="lisp_math",
    steps=10,
    eval_every=5,
    hidden_size=32,
    state_shape=(2, 3, 4),
)

result = train_model(config)
print(result.train_metrics[-1])
```

## Run artifacts

The repository already stores historical experiment outputs under `runs/`. A run may
include:

- `config.json`
- checkpoint files such as `checkpoint_step_000500.pt`
- logs captured externally
- `residual_diagnostics.json`
- summary JSON files for experiment phases

## Related docs

- [`docs/README.md`](./docs/README.md): documentation index
- [`docs/training.md`](./docs/training.md): practical training and experiment guide
- [`docs/reciprocator.md`](./docs/reciprocator.md): formal architecture and math
- [`docs/memory-engines.md`](./docs/memory-engines.md): conceptual background
- [`docs/rl/`](./docs/rl/): RL mathematical reasoning — test plan, lab book, and design docs
- [`lab-book.md`](./lab-book.md): experiment history and conclusions
- [`test-plan.md`](./test_plan.md): experiment matrix and evaluation protocol
- [`corpora/chinese_classics/README.md`](./corpora/chinese_classics/README.md)
- [`corpora/greek_classics/README.md`](./corpora/greek_classics/README.md)
- [`corpora/lisp_math/README.md`](./corpora/lisp_math/README.md)

## Status

This is an active experimental repository. Interfaces are small and readable, but
they are still research code: expect flags and behaviors to evolve as new runs are
added.
