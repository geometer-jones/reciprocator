# RL Mathematical Reasoning Test Plan

## Objective

Teach the reciprocator mathematical and symbolic reasoning through RL on a custom Lisp
dialect. The model learns to produce s-expressions that evaluate to correct results,
progressing through a constructive curriculum: arithmetic → local bindings → code-as-data
→ conditionals/functions → symbolic algebra → proof construction → collections.

Each stage must be demonstrably learnable before the next is added. The curriculum is not
arbitrary — the ordering is determined by constructive necessity: you cannot compose
functions without first being able to name intermediates, and you cannot manipulate
expressions symbolically without first understanding that code is data.

This plan is the prospective companion to `docs/rl-lab-book.md`. The architecture
optimization work in `docs/test-plan.md` is a separate experimental frame; this plan
references its results but does not extend its phase numbering.

## Reporting metrics

**Primary:** accuracy by curriculum stage — fraction of problems where the model's output
parses, evaluates without error, and produces the correct result.

**Secondary:**

- **Reward curve:** mean reward per training batch, broken down by stage and error type
  (correct / syntactically valid but wrong / parse error / eval error / garbage).
- **Curriculum progression:** step at which each stage is unlocked, stability of the
  transition (does accuracy on earlier stages collapse when a new stage is introduced?).
- **Phase-trajectory statistics:** mean and variance of hidden-state phase across the
  output sequence, logged per batch. Hypothesis: correct derivations produce smooth phase
  trajectories; garbage produces chaotic ones. This is exploratory — not a promotion gate
  until Phase R10 validates the signal.
- **Generation samples:** at every eval checkpoint, sample 4 solutions at each active
  stage. Log the prompt, the model's output, the expected result, and whether it was
  correct. Manual inspection for mode collapse and repetitive failure patterns.

## Base architecture

The RL experiments use the reciprocator architecture with configuration drawn from the
best-known recipes in `docs/test-plan.md`. The starting point:

| Parameter               | Value                          | Source                     |
|-------------------------|--------------------------------|----------------------------|
| token_magnitude_type    | inverse_frequency_learned      | test-plan Phase 2 winner   |
| phase_type              | rope                           | test-plan Phase 2 winner   |
| token_phase             | semantic                       | test-plan Phase 2 winner   |
| coupling_type           | sequential                     | test-plan Phase 2S result  |
| normalization_type      | frobenius                      | test-plan Phase 5-6 result |
| readout_type            | phase_aware                   | test-plan Phase 4 result   |
| enable_self_relation    | False                          | not yet validated          |
| num_layers              | 1                              | start minimal              |
| learning_rate           | 1e-3                           | test-plan Phase 7 result   |
| lr_warmup_steps         | 0                              | test-plan Phase 7 result   |
| lr_decay_style          | constant                       | test-plan Phase 7 result   |
| grad_clip_norm          | None                           | test-plan Phase 7 result   |
| weight_decay            | 0.0                            | test-plan Phase 7 result   |

RL-specific hyperparameters (initial defaults, to be tuned):

| Parameter              | Value   | Notes                                         |
|------------------------|---------|-----------------------------------------------|
| grpo_group_size        | 4       | Solutions sampled per problem                 |
| grpo_clip_ratio        | 0.2     | PPO-style clipping                            |
| kl_coeff               | 0.01    | KL penalty against reference policy           |
| reward_temperature     | 1.0     | Softmax temperature for group-relative scores |
| max_output_len         | 32      | Characters — start short                      |
| train_batch_problems   | 32      | Problems per gradient step                    |
| eval_batch_problems    | 64      | Problems per eval checkpoint                  |
| eval_every             | 50      | Steps between evals                           |
| steps                  | 2000    | Per phase                                     |

## Model sizing

Start small. The model must learn arithmetic before anything else matters.

| Phase  | hidden_size | state_shape | Vocab  | Notes                     |
|--------|-------------|-------------|--------|---------------------------|
| R0-R6  | 32          | (2, 3, 4)   | ~30    | Minimal. Iterate fast.    |
| R7-R9  | 64          | (4, 4, 4)   | ~50    | Scale for harder stages.  |
| R10-R12| 128         | (4, 4, 4)   | ~50    | Phase-trajectory analysis |

The vocabulary is the set of unique characters that appear in s-expressions plus
problem/solution formatting tokens. Expected: digits `0-9`, operators `+-*/`, parens
`()`, brackets `[]`, braces `{}`, colon `:`, space, newline, and a handful of symbol
characters. Character-level tokenizer throughout — no subword, no special tokens.

## Curriculum stages

The stages are ordered by constructive necessity. Each stage introduces new forms that
are not constructible from the previous stages alone.

### Stage 1: Arithmetic

Forms: `+`, `-`, `*`, `/`, `mod`, `expt`
Inputs: numeric expressions
Outputs: a single number

```lisp
; prompt: (+ 3 (* 4 5))
; expected output: 23
```

Problem parameters: depth (1-4), value range (0-99 for depth 1, 0-9 for depth 4),
operator set.

### Stage 2: Let and comparison

Forms: `let`, `=`, `<`, `>`, `<=`, `>=`
Inputs: expressions with local bindings
Outputs: a single number or boolean

```lisp
; prompt: (let [x (+ 3 4)] (* x 2))
; expected output: 14
```

`let` is the minimal form for multi-step reasoning — name an intermediate result, then
use it. Without it, the model can only produce single-expression evaluations.

### Stage 3: Quote and code-as-data

Forms: `quote`, `list`, `eval`
Inputs: expressions involving quoted data
Outputs: a value or a list structure

```lisp
; prompt: (eval (quote (+ 1 2)))
; expected output: 3
; prompt: (first (quote (a b c)))
; expected output: a
```

This is the metacomputation transition. The model must understand that s-expressions are
data structures that can be operated on, not just things to be evaluated. `quote` suspends
evaluation; `eval` resumes it. Without this stage, the model cannot move to symbolic
manipulation.

### Stage 4: Conditionals and functions

Forms: `if`, `cond`, `define`, `lambda`
Inputs: expressions with branching and/or function definition
Outputs: a value

```lisp
; prompt: (if (> 5 3) (+ 1 1) (* 2 2))
; expected output: 2
; prompt: ((lambda [x] (* x x)) 7)
; expected output: 49
```

Functions require `let` (stage 2) for parameter binding and the code-as-data insight
(stage 3) for `lambda` being a value. This is why stages are ordered this way.

### Stage 5: Symbolic algebra

Forms: `simplify`, `expand`, `substitute`, `differentiate`
Inputs: expressions containing symbolic variables
Outputs: a simplified/transformed expression (not a number)

```lisp
; prompt: (simplify (+ (* 2 x) (* 3 x)))
; expected output: (* 5 x)
```

Reward: symbolic equivalence checked by (a) structural normalization and comparison,
and (b) numeric spot-checks — substitute random values for free variables in both the
model's output and the expected result, verify they agree across 10+ samples.

This stage requires code-as-data (stage 3) because the model must manipulate expression
structure, not just evaluate it.

### Stage 6: Proof construction

Forms: `(step rule-name input output)` sequences, plus a known rule set
Inputs: `(prove target premises)` where `target` is an expression and `premises` is a
list of equations or axioms
Outputs: a sequence of steps, each applying a known rule

```lisp
; prompt: (prove (* 6 x)
;          (from (* 2 x) (* 3 x)))
; expected output:
; (step collect-like-terms (+ (* 2 x) (* 3 x)) (* 5 x))
; ... etc
```

Reward: fraction of valid steps. Each step is checked individually — the rule must be
known, the input must match the rule's pattern, and the output must be the correct
application. Full credit for complete valid proof; partial credit proportional to valid
step fraction.

This stage requires symbolic algebra (stage 5) because the model must understand
expression transformation as a first-class operation.

### Stage 7: Collections

Forms: vectors `[1 2 3]`, maps `{:a 1}`, `get`, `count`, `conj`, `assoc`, `mapv`,
`filterv`, `reduce`, `->`, `->>`, `comp`, `partial`
Inputs: expressions involving collection operations
Outputs: a value or collection

```lisp
; prompt: (-> [1 2 3 4 5] (filterv odd?) count)
; expected output: 3
```

This is the widest stage. It can be split into sub-stages (vectors first, then maps,
then higher-order operations) if the curriculum controller needs finer granularity.

---

## Fixed hyperparameters for RL phases

| Parameter              | Value   | Notes                                         |
|------------------------|---------|-----------------------------------------------|
| curriculum_promote     | 0.80    | Promote when stage accuracy exceeds this       |
| curriculum_demote      | 0.30    | Demote when stage accuracy drops below this    |
| curriculum_mix_ratio   | 0.70    | Fraction of problems from current stage        |
| partial_credit_valid   | 0.50    | Syntactically valid, wrong value               |
| partial_credit_parse   | 0.20    | Parses but eval errors                         |
| reward_baseline        | mean    | GRPO baseline: mean reward in group            |
| max_generation_retries | 3       | Retry on empty/garbage output                  |

## Evaluator specification

The evaluator (`lisp_eval.py`) is a Python module with a single public interface:

```python
def eval_sexpr(expr: str) -> tuple[Any, str | None]:
    """Parse and evaluate an s-expression string.

    Returns (result, error) where error is None on success.
    Error types: "parse_error", "eval_error", "undefined_form".
    """
```

It handles three bracket types, keywords, and all forms listed in the curriculum stages.
Forms are gated by a `stage` parameter — calling `(lambda ...)` at stage 1 returns an
`"undefined_form"` error, not a crash. The evaluator has no external dependencies.

The evaluator is the fixed point of the system. Its semantics define what "correct"
means. Changes to the evaluator require re-baselining all prior results.

---

## Phases

### Phase R0: Evaluator and problem generator validation

**No training.** Verify infrastructure.

**Tests:**

1. Evaluator correctly parses and evaluates all forms at each stage.
2. Evaluator returns correct error types for malformed input.
3. Evaluator rejects forms from later stages when called at an earlier stage.
4. Problem generator produces syntactically valid problems at every parameterization.
5. Every generated problem's expected result matches an independent evaluation of the
   prompt expression (cross-check: generate 1000 problems per stage, evaluate the prompt,
   verify agreement with the expected result).
6. Reward function returns 1.0 for correct answers, 0.5 for valid-but-wrong, 0.2 for
   parseable-but-eval-error, 0.0 for garbage — on hand-crafted test cases at each stage.

**Decision rule:** proceed only when all six checks pass. If the evaluator and problem
generator disagree on any case, fix the evaluator — it is the ground truth.

---

### Phase R1: GRPO smoke test

**3 runs** (1 config × 3 seeds). Steps = 200. Stage 1 only (arithmetic, depth 1).

Config: `hidden_size=32`, `state_shape=(2,3,4)`, `grpo_group_size=4`.

Purpose: verify the RL training loop produces gradients, the model's output changes over
training, and the reward signal flows correctly. Not expecting mastery — just movement.

**Checks:**

- Gradient norms are finite and non-zero at every step.
- Mean reward changes by > 0.1 from start to end.
- At least some generated outputs are parseable s-expressions by step 200.
- No NaN/Inf in model parameters.

**Decision rule:** if any check fails, debug the RL loop before proceeding. A working
gradient path is the prerequisite for everything else.

---

### Phase R2: Arithmetic mastery

**6 runs** (2 configs × 3 seeds). Steps = 2000.

The first real test: can the model learn `(+ 1 2)` → `3`?

| Run | Config variation             | What it tests                          |
|-----|-----------------------------|----------------------------------------|
| R2a | depth 1, operators `+ -`    | Minimal arithmetic                    |
| R2b | depth 2, operators `+ - *` | Deeper expressions, more operators    |

**Reporting:** accuracy at eval checkpoints, reward curve, sample outputs at steps
{100, 500, 1000, 2000}.

**Decision rule:** if R2a reaches > 80% accuracy by step 2000, the model can learn
arithmetic. If R2b also reaches > 70%, proceed to Phase R3 with `depth ≤ 2` and all
four operators. If R2a does not reach 80%, the model or RL loop needs debugging before
adding complexity.

**Failure modes to watch:**

- Model produces only one number regardless of input (mode collapse).
- Model learns to parse but always evaluates to 0 (hasn't learned arithmetic).
- Reward plateaus below 0.5 (partially parseable but never correct).

---

### Phase R3: Curriculum controller validation

**4 runs** (2 configs × 2 seeds). Steps = 3000.

Tests the curriculum controller's ability to promote/demote between stages 1 and 2.

| Run | Stages | What it tests                               |
|-----|--------|---------------------------------------------|
| R3a | 1 → 2  | Normal promotion: arithmetic then let       |
| R3b | 2 only | Control: start at stage 2 without stage 1   |

**Reporting:** stage unlock step, accuracy on each stage over time, whether accuracy on
stage 1 degrades after stage 2 is unlocked.

**Decision rules:**

- If R3a promotes to stage 2 and maintains > 70% accuracy on stage 1: curriculum
  controller works. Proceed to Phase R4.
- If accuracy on stage 1 collapses after promotion: the curriculum mix ratio is too
  aggressive. Reduce `curriculum_mix_ratio` to 0.50 and rerun.
- If R3b reaches stage 2 competence without stage 1: stage ordering is not
  constructive-necessity at this level — the model can skip. Document this; it means
  the stage ordering assumption is weaker than expected for arithmetic/let.
- If R3b fails and R3a succeeds: stage ordering matters. The constructive hierarchy
  is real.

---

### Phase R4: Multi-stage progression

**4 runs** (2 configs × 2 seeds). Steps = 5000 per run.

Tests whether the model can progress through stages 1-4 (arithmetic through
conditionals/functions). This is the core curriculum test.

| Run | Stages | hidden_size | What it tests                         |
|-----|--------|-------------|---------------------------------------|
| R4a | 1 → 4  | 32          | Full progression at minimal size      |
| R4b | 1 → 4  | 64          | Does more capacity help progression?  |

**Reporting:** step at which each stage is unlocked, accuracy per stage over time, total
stages reached, sample outputs at each stage transition.

**Decision rules:**

- If R4a reaches stage 3 (quote): the model can learn the code-as-data transition.
  This is the key result — it means the model can cross from computation to
  metacomputation.
- If R4a reaches stage 4: the full constructive hierarchy through functions is learnable.
  Proceed to Phase R5.
- If R4a stalls at stage 2: `let` is learnable but `quote` is not at this model size.
  Try R4b. If R4b also stalls, the architecture may need modification (e.g., phase-aware
  readout to preserve structural information).
- If R4b progresses further than R4a: model size is a bottleneck. Note the minimum
  viable size for each stage.

---

### Phase R5: Symbolic algebra

**Conditional:** run only if Phase R4 reaches stage 3 (quote).

**4 runs** (2 configs × 2 seeds). Steps = 5000.

| Run | hidden_size | What it tests                             |
|-----|-------------|-------------------------------------------|
| R5a | 64          | Symbolic algebra at medium size           |
| R5b | 128         | Symbolic algebra at larger size           |

Problem difficulty: expressions with 1-3 free variables, depth 2-4, operations from the
`simplify` and `expand` families.

**Reporting:** accuracy (symbolic equivalence), accuracy on numeric spot-checks (partial
credit), sample transformations.

**Decision rules:**

- If R5a reaches > 60% symbolic equivalence: symbolic manipulation is learnable. Proceed
  to Phase R6.
- If R5a fails but numeric spot-checks are > 70%: the model is producing numerically
  equivalent but structurally different expressions. Consider normalizing before
  comparison.
- If both fail: the model cannot yet manipulate expression structure. Return to Phase R4
  and verify stage 3 mastery is solid before retrying.

---

### Phase R6: Proof construction

**Conditional:** run only if Phase R5 reaches > 60% symbolic equivalence.

**4 runs** (2 configs × 2 seeds). Steps = 5000.

| Run | hidden_size | What it tests                             |
|-----|-------------|-------------------------------------------|
| R6a | 128         | Proof construction                        |
| R6b | 128         | Proof construction with phase-trajectory aux reward |

R6b adds an auxiliary reward based on phase-trajectory smoothness (see Phase R10). This
tests whether the phase signal helps before doing a dedicated analysis.

Problem difficulty: proofs requiring 2-6 steps, rule set limited to algebraic
transformations (collect-like-terms, distribute, substitute, simplify).

**Reporting:** valid step fraction, complete-proof fraction, sample proofs with step-level
annotations.

**Decision rules:**

- If R6a reaches > 50% complete proofs: proof construction is learnable at this scale.
- If R6b beats R6a by > 10% on complete-proof fraction: the phase-trajectory signal is
  useful. Promote it to a standard part of the reward.
- If neither reaches 30%: proof construction is too hard at current scale. Document and
  defer to a scaling phase.

---

### Phase R7: Collections

**Conditional:** run only if Phase R4 reaches stage 4.

**4 runs** (2 configs × 2 seeds). Steps = 3000.

Tests vectors, maps, and collection operations as curriculum stage 7.

| Run | Sub-stages              | What it tests                              |
|-----|-------------------------|--------------------------------------------|
| R7a | vectors → maps → ops    | Gradual introduction of collection types   |
| R7b | all at once             | Can the model learn collections without sub-staging? |

**Reporting:** accuracy by sub-stage, error types (bracket mismatch, wrong collection
type, wrong operation).

**Decision rule:** if R7a reaches > 70% on vectors and > 50% on collection operations,
collections are learnable. If R7b also succeeds, sub-staging is unnecessary for this
stage.

---

### Phase R8: RL hyperparameter sweep

**8 runs** (4 configs × 2 seeds). Steps = 3000. Stage 1 only (arithmetic).

Now that the loop works, tune the RL-specific hyperparameters.

| Run | grpo_group_size | kl_coeff | max_output_len | What it tests                   |
|-----|-----------------|----------|----------------|---------------------------------|
| R8a | 2               | 0.01     | 32             | Minimal group                   |
| R8b | 8               | 0.01     | 32             | Larger group (lower variance)   |
| R8c | 4               | 0.001    | 32             | Weaker KL constraint            |
| R8d | 4               | 0.01     | 64             | Longer output (deeper exprs)    |

**Reporting:** accuracy at step 3000, reward curve, sample outputs, wall-clock time.

**Decision rule:** adopt the config with the best step-3000 accuracy. If multiple configs
are within 2%, prefer the one with lower wall-clock time. Document the sensitivity — if
`grpo_group_size` matters a lot, the variance in group-relative scoring is high and the
group should be increased in later phases.

---

### Phase R9: Curriculum robustness

**6 runs** (3 configs × 2 seeds). Steps = 5000.

Tests whether the curriculum controller is robust to hyperparameter choices.

| Run | promote | demote | mix_ratio | What it tests                          |
|-----|---------|--------|-----------|----------------------------------------|
| R9a | 0.80    | 0.30   | 0.70      | Default                                |
| R9b | 0.70    | 0.20   | 0.50      | More aggressive promotion, more mixing |
| R9c | 0.90    | 0.40   | 0.85      | Very conservative, little mixing       |

**Reporting:** stages reached, total steps to reach each stage, accuracy stability at
each stage, catastrophic forgetting events (accuracy on earlier stages dropping below
demote threshold).

**Decision rule:** if all three configs reach the same final stage, the curriculum is
robust to these settings — use the fastest one. If they diverge, the controller is
sensitive and needs careful tuning before scaling.

---

### Phase R10: Phase-trajectory analysis

**No new training.** Post-hoc analysis on checkpoints from Phases R2-R6.

At every saved checkpoint, for every eval sample:

1. Run a forward pass, recording the complex-valued hidden state at each output position.
2. Compute phase variance across the output sequence: `var(angle(hidden))`.
3. Compute phase autocorrelation at lag 1.
4. Bin samples by reward (1.0 / 0.5 / 0.2 / 0.0) and compare distributions.

**Hypothesis:** correct outputs (reward 1.0) have lower phase variance and higher phase
autocorrelation than incorrect outputs. If this holds, the phase trajectory is a proxy
for "the model is constructing a valid solution" and can be used as an auxiliary reward.

**Reporting:** phase variance distributions by reward bucket, ROC curve for
correct/incorrect classification from phase statistics alone, correlation between phase
variance and step-level accuracy within proof-construction outputs.

**Decision rule:**

- If phase variance separates correct from incorrect with AUC > 0.7: the signal is real.
  Add it as an auxiliary reward in future phases. Proceed to Phase R11 to test whether
  it helps training.
- If AUC is 0.5-0.7: weak signal. Log it, but do not add as reward — the noise would
  hurt more than the signal helps.
- If AUC < 0.5: the signal is absent or inverted. The complex-valued state's phase is
  not encoding computational traces in a detectable way at this scale.

---

### Phase R11: Phase-auxiliary reward

**Conditional:** run only if Phase R10 finds AUC > 0.7.

**4 runs** (2 configs × 2 seeds). Steps = 3000. Stage 1 only (arithmetic).

| Run | reward                           | What it tests                              |
|-----|----------------------------------|--------------------------------------------|
| R11a| outcome only (standard)          | Baseline                                   |
| R11b| outcome + phase-trajectory aux   | Does the phase signal accelerate learning? |

The auxiliary reward is: `phase_bonus = max(0, 1 - phase_variance / baseline_variance)`,
scaled by a coefficient (start at 0.1). Total reward = outcome_reward + phase_bonus.

**Decision rule:** if R11b reaches 80% accuracy in fewer steps than R11a, the phase
signal is useful for training, not just for post-hoc analysis. Promote to standard
reward component. If R11b is slower or equivalent, the phase signal is diagnostic but
not actionable.

---

### Phase R12: Scaling

**Conditional:** run only if Phases R2-R4 establish that the curriculum works at minimal
size.

**4 runs** (2 sizes × 2 seeds). Steps = 5000. Stages 1-4.

| Run  | hidden_size | state_shape | What it tests                     |
|------|-------------|-------------|-----------------------------------|
| R12a | 64          | (4, 4, 4)   | Medium scale                      |
| R12b | 128         | (4, 4, 4)   | Larger scale                      |

**Reporting:** stages reached, accuracy per stage, learning speed (steps to 80% per
stage), wall-clock time.

**Decision rule:** if R12b reaches more stages or higher accuracy than R12a, the
architecture benefits from scale. If R12a and R12b reach the same stages with similar
accuracy, the curriculum — not model capacity — is the bottleneck. Focus effort on
problem generation and reward shaping rather than model size.

---

### Phase R13: Architecture comparison

**Conditional:** run only if Phase R12 completes.

Compares the reciprocator against a standard GRPO-trained character-level transformer
on the same curriculum.

**6 runs** (2 architectures × 3 seeds). Steps = 5000. Stages 1-4.

| Run  | Architecture              | hidden_size | What it tests                    |
|------|---------------------------|-------------|----------------------------------|
| R13a | Reciprocator (R12 winner) | [from R12]  | Baseline                         |
| R13b | Transformer (param-matched) | [matched] | Does the complex state help?     |

The transformer baseline: standard pre-norm causal transformer with RoPE, SwiGLU FFN,
character-level tokenizer, same vocab. Parameter-matched within 5%. Same RL loop, same
curriculum, same problem generator.

**Reporting:** accuracy by stage, learning speed, total stages reached. If Phase R10
found a phase signal, report whether the transformer's hidden state shows analogous
structure.

**Decision rule:**

- If the reciprocator reaches more stages or higher accuracy: the complex-valued state
  is contributing to mathematical reasoning. Document why.
- If the transformer wins: the complex-valued state is not earning its complexity for
  this task. Report honestly. The reciprocator's advantage (if any) lies elsewhere.
- If they tie: the task is not discriminating. Either increase difficulty or accept that
  architecture choice doesn't matter at this scale for this task.

---

## Run summary

| Phase | Purpose                              | Configs | Seeds | Runs  | Status      |
|-------|--------------------------------------|---------|-------|-------|-------------|
| R0    | Evaluator + generator validation     | n/a     | n/a   | 0     | Passed 2026-04-23 |
| R1    | GRPO smoke test                      | 1       | 3     | 3     | Diagnostic passed 2026-04-23 |
| R2    | Arithmetic mastery                   | 2       | 3     | 6     |             |
| R3    | Curriculum controller validation     | 2       | 2     | 4     |             |
| R4    | Multi-stage progression (1-4)        | 2       | 2     | 4     |             |
| R5    | Symbolic algebra                     | 2       | 2     | 4     |             |
| R6    | Proof construction                   | 2       | 2     | 4     |             |
| R7    | Collections                          | 2       | 2     | 4     |             |
| R8    | RL hyperparameter sweep              | 4       | 2     | 8     |             |
| R9    | Curriculum robustness                | 3       | 2     | 6     |             |
| R10   | Phase-trajectory analysis (post-hoc) | n/a     | n/a   | 0     |             |
| R11   | Phase-auxiliary reward               | 2       | 2     | 4     |             |
| R12   | Scaling                              | 2       | 2     | 4     |             |
| R13   | Architecture comparison              | 2       | 3     | 6     |             |
| **Total** |                                  |         |       | **57** |             |

---

## Dependencies

```
R0 → R1 → R2 → R3 → R4 → R5 → R6
                     ↘ R7
         ↘ R8 (independent, can run after R2)
         ↘ R9 (independent, can run after R4)
R2-R6 → R10 → R11
R4 → R12 → R13
```

Phases R8 and R9 are independent of the curriculum progression and can run in parallel
once R2 establishes a working RL loop. Phase R10 depends on checkpoints from R2-R6.
Phase R11 depends on R10's result. Phase R12 depends on R4. Phase R13 depends on R12.

---

## Relationship to `docs/test-plan.md`

The existing test plan optimizes the reciprocator architecture for language modeling
quality (val_bpc) under supervised training on text corpora. This plan optimizes the
same architecture for mathematical reasoning under RL training on generated problems.

The two plans share:

- The model architecture (`ReciprocatorLM`)
- The training infrastructure (`training.py`)
- The character-level tokenizer

They differ in:

- Training objective (cross-entropy vs. GRPO reward)
- Data source (bundled corpora vs. generated problems)
- Evaluation metric (val_bpc vs. accuracy by stage)
- Success criteria (language modeling quality vs. mathematical reasoning capability)

If the existing test plan's later phases (11-13) produce a significantly different
architecture, the base architecture in this plan should be updated to match. The RL
experiments are consumers of the architecture optimization work, not drivers of it.

---

## Analysis notes

- The curriculum ordering is a hypothesis, not a fact. Phase R3 (and specifically the
  R3b control) tests whether the ordering matters. If stages are independently learnable,
  the constructive-necessity framing is wrong and the curriculum controller can be
  simpler.
- The evaluator is the ground truth. Any discrepancy between the evaluator and the
  problem generator's expected results is a bug in the generator, not the evaluator.
- Reward shaping matters more at early stages when the model is producing mostly garbage.
  A reward of 0.2 for parseable output gives the model a gradient signal before it can
  produce correct answers. If the model never gets past garbage, the partial-credit
  coefficients may need adjustment.
- The phase-trajectory hypothesis (Phase R10) is the most architecturally novel claim
  in this plan. If it fails, the RL pipeline still works — it just doesn't exploit the
  complex-valued state in a way that a real-valued model couldn't.
- Report wall-clock time for every phase. RL is slower than supervised training (multiple
  samples per problem, evaluation at each step). If a phase takes > 10x the wall time of
  a comparable supervised run, the RL loop needs optimization before scaling.
- Catastrophic forgetting is the primary risk at stage transitions. If the model forgets
  arithmetic when it learns `let`, the curriculum mix ratio or the KL penalty is too low.
  Log accuracy on all unlocked stages at every eval, not just the current stage.
- The character-level tokenizer is a deliberate constraint. It forces the model to learn
  the structure of s-expressions from raw characters. If the model cannot parse `(+ 1 2)`
  after 2000 steps, consider whether character-level is too hard at this model size
  before blaming the RL loop.
