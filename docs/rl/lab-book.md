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

_No phases have been run yet._

## Current Conclusions

_None yet._
