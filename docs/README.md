# Documentation

This directory collects the longer-form documentation for `reciprocator`.

## Start here

- [`../README.md`](../README.md): repository overview, installation, quick start,
  and layout.
- [`training.md`](./training.md): practical guide to running experiments with
  `scripts/train.py`, choosing important flags, and understanding outputs.

## Conceptual and mathematical docs

- [`reciprocator.md`](./reciprocator.md): formal architecture and mathematical
  specification of the Reciprocator.
- [`memory-engines.md`](./memory-engines.md): higher-level geometric intuition
  and motivation behind the architecture.

## Experiment records

- [`../test_plan.md`](../test_plan.md): planned experiment matrix and evaluation
  criteria.
- [`../lab-book.md`](../lab-book.md): retrospective log of what actually ran and
  what conclusions were drawn from those runs.

## Data docs

- [`../corpora/plato_jowett/README.md`](../corpora/plato_jowett/README.md):
  bundled Plato corpus details.
- [`../corpora/greek_philosophy_classics/README.md`](../corpora/greek_philosophy_classics/README.md):
  bundled Greek philosophy corpus details.

## Implementation map

If you are reading the code, these are the main entry points:

- `scripts/train.py`: training CLI
- `src/reciprocator/training.py`: dataset building, training loop, evaluation,
  benchmarking, checkpoint logic
- `src/reciprocator/model.py`: token lift, blocks, optional attention, LM wrapper
- `src/reciprocator/mixer.py`: reciprocator tensor-state mixer and coupling backends
- `src/reciprocator/complex_ops.py`: complex-valued normalization and layers
- `src/reciprocator/corpora.py`: bundled corpus registry and file access helpers
