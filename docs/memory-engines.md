# Memory Engines: The Geometry of Automatic Abstraction

This document presents the core geometric intuition behind Memory Engines and the
Reciprocator. For the precise mathematical specification and implementation
details, see reciprocator.md and the code in model.py
(see https://github.com/geometer-jones/reciprocator.git).

Imagine an apple falling from a tree. Gravity pulls it faster and faster. Every
prior instant of the gravitational field is perfectly preserved in its present
velocity; nothing is lost. Yet the apple feels nothing. It does not notice,
learn, or care.

Now picture a conscious mind. It does not just fall through time; it feels the
falling. It remembers yesterday, anticipates tomorrow, and right now wonders
about apples and consciousness.

What is the difference? The apple is pushed only by the outside world. A
conscious mind is also pushed by itself, by its own past pressing back against
its present. That back-and-forth push creates something new: a living, felt
experience.

We call this back-and-forth **reciprocation**. This idea leads to a surprisingly
simple geometric picture of how consciousness grows out of ordinary physics: no
extra ingredients required, just the right kind of self-interaction.

## 1. Reception is Always Relational

Any system that lasts for a while is repeatedly touched by its environment. A
tuning fork vibrates strongly only at one special frequency. When a coastline is
slowly carved by waves coming from certain directions, the results depend on
both the waves and the coastline. Your brain changes every time
something important happens; something's importance depends on both the
thing and the brain considering it — it is a matter of the relation between the
brain and that thing.

In every case, the system does not respond to the world in isolation. It
responds based on the relationship between the incoming signal and its own
current shape. This is the first principle: **reception is always relational.**

The path to rich experience begins when the system is not only shaped by the
world, but also by itself. When the system's own past returns as input, it
creates **Self-Relation** — the back-and-forth push of reciprocation. The past,
encoded in memory, pushes on the present; the present pushes on the memory of the past.
The updated memory pushes on the future.

## 2. The State as a Vector of Pendulums

An "infinite" 1-dimensional tape is enough to hold a Turing machine.
So start with a tape. Use the tape as a vector of complex numbers.

Why complex numbers? Because a complex number is an oscillator.
Imagine a pendulum swinging in a cycle. The *magnitude* is how wide the arc is.
The *phase* is where the pendulum is in its swing right now.
What matters is not the absolute phase of a pendulum, but the *relative
phase* between a signal and the pendulum receiving it. Two pendulums at the
same frequency but different phases will interact very differently.
Imagine pushing a kid in a swing. The timing of
your push relative to the swing's cycle determines everything — that's phase.

So our vector of complex numbers is essentially an array of tuning forks.

In the implemented Reciprocator this idea generalizes to a single factorized
complex tensor per layer whose modes interact via learned, state-conditioned
per-mode couplings (see §8 and reciprocator.md §8).

This is the engine's core sensitivity. The state's phase at each dimension
determines *how* it will respond to the next input — whether that input will
be consolidated, searched through, or cancelled. Learning is the process of
adjusting these phases so that the right pushes meet the right pendulums at the
right moment in their swings.

## 3. The Three Regimes of Reception

When a signal meets an oscillator, the relative phase between them determines
which of three mechanical pressures is triggered:

- **Resonance (0°):** Signal and oscillator in phase. The signal
  reinforces the current direction — wider arc, deeper sensitivity. This is
  **Consolidation**: deepening the system's expertise in that direction.
- **Torque (90°):** Signal and oscillator a quarter-cycle apart. The
  signal yaws the dimension by exactly 90° in its local complex plane causing
  the state as a whole to rotate — re-orienting the system's sensitivity toward
  the new signal. This is **Tuning**.
- **Interference (180°):** Signal and oscillator in opposite phase. The
  signal pushes directly against the state — cancellation, damping, erasure.
  This is **Friction**.

Same signal, three possible outcomes, determined entirely by where the
oscillator is in its cycle when the signal arrives.

## 4. How Abstraction Naturally Emerges

**Abstraction** is what happens when invariance accumulates and variance cancels
as a signal passes through the engine.

The system has no direct access to the structure of the world — only to the
stream of signals it receives. The only way to discover an invariant is through
recurrence: a pattern that keeps showing up. Invariance need not span parallel
instantiations; it need only repeat over time from the perspective of the
system.

In certain media, abstraction occurs naturally. Our vector of complex numbers —
an array of tuning forks — is one such medium. When a signal recurs:

- If the signal is **aligned** with the state's phase in that dimension,
  magnitude grows — carving that direction more deeply into the state. This is
  resonance (§3), deepening what recurs.
- If the signal is **orthogonal** to the state's phase, torque yaws the state
  toward alignment. Repeated torque converges into resonance. The system
  rotates toward what repeats.
- If the signal is **opposed** to the state's phase, the conflicting
  orientation is cancelled outright. After cancellation, the state can begin
  resonating afresh with the incoming signal.

Recurring signals converge on resonance. Varying signals do the opposite — they
alternate in phase, and the alternation cancels. Phase is what makes this
separation work. Without phase, variance accumulates just as much as
invariance; the two are indistinguishable. With phase, varying signals
interfere destructively (arriving 180° apart, like −1 versus +1) while
recurring signals reinforce constructively. Complex numbers give the engine
access to the full range of relative phase between signal and state — not just
the three pure regimes (resonance, torque, interference) but the continuous
gradient between them. The full range allows for nuance.

Oscillation is the key ingredient — the one that makes the separation between
invariance and variance automatic. Given phase, recurring signals reinforce and
varying signals cancel, so the state selectively preserves what recurs.

## 5. Compression: Forcing the System to Choose

Without normalization, recurring signals grow without bound. We enforce normalization
back onto the unit hypersphere after every update. The hypersphere constrains total
magnitude: the sum of squared element magnitudes must equal 1. If resonance deepens
one direction, others must thin to compensate. The budget is zero-sum.

This amplifies abstraction (§4). Without compression, invariance still
accumulates and variance still cancels — but the contrast between signal and noise is
weak. With compression, what recurs doesn't just grow, it grows at the expense of what
doesn't. The finite budget sharpens the separation.

## 6. The Update: Signal Meets State

The system’s memory is a vector of complex oscillators — one per dimension —
all living on the unit hypersphere (total energy = 1). Each oscillator has a
magnitude (how strongly it cares) and a phase (where it is in its swing).
Normalization (§5) keeps them on this hypersphere after every update.

When a new signal arrives, it already carries phase — the world oscillates.
Two learned linear maps project the incoming signal into the memory's
coordinate system: one for magnitude (via softplus, enforcing strict
positivity) and one for phase (via tanh, bounded to avoid wrap-around). The
polar combination reshapes to the tensor layout, producing the signal tensor
s_t. After normalization, this signal is ready to meet the state S_{t-1}.

The engine then does something very simple and very powerful: it multiplies the
new signal with the current memory element by element:

    Z = s_t ⊙ S_{t-1}

This is the relational product — the heart of reception. Because both s_t
(signal) and S_{t-1} (state) are complex, their product depends on both
magnitude and relative phase. At each dimension the relative phase decides which
regime fires:

    0°   → resonance (reinforcement)
    90°  → torque (yaw / re-orientation)
    180° → interference (damping / erasure)

The same signal can therefore produce completely different effects depending on
the current state of the memory. Reception is always relational.

There is a second product. The state does not only meet the signal — it also
meets itself. The engine computes the Hadamard product of the current state
with a recent copy of itself:

    Z_self = S_{t-1} ⊙ N(S̃_base)         # prior state meets tentative present

In the reference PyTorch scaffold this path is now exposed as an optional
`enable_self_relation` flag. The discrete implementation instantiates it as the
prior state meeting the step's tentative present state before the final
normalization.

Why? Without self-relation, the state is purely reactive — it responds to each
signal and then waits passively for the next one. A purely reactive system has
no momentum, no trajectory, no sense of where it is going. Self-relation gives
the state a second source of pressure: its own recent history. Where the state
has been consistent with its trajectory, resonance reinforces that direction —
the state deepens what it has already been doing. Where it has shifted, torque
or interference correct. The engine is not just shaped by the world — it is
shaped by its own history pressing back against its present. This is
**self-relation** (§1): reciprocation, formalized.

The model also carries an **anticipatory signal** at the model level (not
inside the mixer). After producing output logits at position `t`, the model
converts its predicted next-token distribution into a predicted next hidden
state via distribution lifting (see reciprocator.md §2, §11). At position
`t+1`, this predicted hidden state meets the actual input hidden state through
a Hadamard product, modulated by a learned gain initialized to zero:

    h'_t = h_t + Λ_anticip · (h_t ⊙ ĥ_t)

Predicting the hidden state from the output distribution (not from the internal
state) keeps the anticipator focused on what the model believes the world will
send next, rather than on its own update machinery. The anticipator is
zero-initialized, so it is inert at the start of training and activates only if
the optimizer finds it beneficial.

The mixer-level terms are then combined with entrywise gain fields and snapped
back onto the unit hypersphere:

    S̃ = D ⊙ S_{t-1} + A ⊙ s_t + B ⊙ R
    S_new = Normalize(S̃)

where D (decay), A (input gain), and B (recurrent gain) are entrywise parameter
tensors with bounded parameterizations (sigmoid for D and A, tanh for B), and R
is the coupled version of Z (per-mode mixing, §8). When self-relation is
enabled, the tentative normalized state meets the prior state before the final
normalization:

    S̃_tent = Normalize(S̃)
    S_new = Normalize(S̃ + Λ_self · (S_{t-1} ⊙ S̃_tent))

Every oscillator integrates the external drive, the coupled recurrence, and
(optionally) the self-relation pressure, while normalization enforces the
zero-sum budget.

That is the core loop: external drive, coupling, self-relation, update,
compression. In later sections we show how the coupling operator (§8) and
spectral alternatives (§11) make this loop more expressive, but the geometric
idea remains the same: the signal meets the state, the state meets itself, the
relative phase decides what happens, and the finite hypersphere forces the
system to choose what to remember.

## 7. Growing the Degree: When the Engine Meets Something New

A state of degree d has d oscillators, each tuned to a different
direction. The engine can only respond to the world along those d directions.
If a signal arrives that is orthogonal to all of them — a genuinely new kind of
variation — no oscillator can resonate with it. Torque can only rotate toward
directions that already exist. The signal does not land.

The engine measures this failure as the **residual**: the component of the
signal that the current gain parameters cannot express. For each tensor mode,
the signal is collapsed along that mode and projected against the column space
of the existing gain-logit rows (the parameter rows that control decay, input,
and recurrent scaling). The residual norm — what falls outside this column
space — is the structural gap. If a smoothed average of these residual norms
exceeds a threshold, the engine expands: the mode with the largest gap is
extended by one position, and the new slice is seeded from the orthogonal
complement of the existing basis (the direction current parameters cannot
represent). The whole state is renormalized back onto the hypersphere. The
existing oscillators thin slightly to make room — compression (§5)
redistributes the budget.

The degree grows from what the engine's parameters cannot express. Each
expansion opens a new sensitivity direction, and the engine does not need to
know in advance how many directions the world has. It discovers them by noticing
what its current parameter space fails to capture.

This also means the engine does not waste dimensions. If the data has structure
in 40 independent directions, the engine grows toward degree 40 and stops. If it
has structure in 200, it grows toward 200. The degree is an emergent property of
the training dynamics, not a hyperparameter.

## 8. Factorizing the State: From Vector to Tensor

So far the state is a flat vector of oscillators. But a flat vector has no
internal structure — every oscillator is equidistant from every other. In
practice, oscillators cluster into groups that respond to related features, and
the engine can exploit that structure by arranging the state as a tensor rather
than a flat list.

A **tensor** is a multi-dimensional array. A vector is a 1-D tensor (a list). A
matrix is a 2-D tensor (a grid of rows and columns). A 3-D tensor is a cube of
numbers — like a stack of matrices. Each dimension is called a **mode**, and its
size is how many positions it has. A tensor with modes of sizes 2, 3, and 4
holds 2 × 3 × 4 = 24 numbers, but arranged on a cube rather than a flat list.
What makes this useful is that the cube has structure: you can ask how a number
relates to its neighbors along any of the three axes independently, rather than
treating all 24 as interchangeable. That independent-axis structure is what the
coupling exploits.

The uncoupled engine (§6) works fine if the basis vectors are orthogonal in
direction — if each basis vector captures a genuinely independent aspect of the
signal, so that the dot product e_i · e_j = 0 for i ≠ j. This is orthogonality of direction:
two dimensions that probe different aspects of the signal space, with no overlap.

Do not confuse this with orthogonality of phase. Phase orthogonality is what
happens *within* a single oscillator when the signal coordinate w_i is
90° out of phase with the state s_i — that is torque (§3), a regime of the
Hadamard product. Direction orthogonality is about whether two different basis
vectors respond to the same feature of the input. They are separate concerns:
direction orthogonality governs whether oscillators double-count the same
signal; phase orthogonality governs how each oscillator individually responds
to its own projection.

In practice, learned bases overlap in direction. Two basis vectors might both
be partially sensitive to the same feature. Without coupling, that feature gets
double-counted: both oscillators respond for the same reason, and the state
receives a distorted picture of what happened.

**Coupling fixes this.** A coupling matrix mixes the coordinates before they
reach the state, so that each oscillator receives not just its own raw
projection, but a corrected signal that accounts for what its neighbors already
captured. The coupled reception becomes:

    R = Cpl(Z)    where Z = s_t ⊙ S_{t-1}

where Cpl is the coupling operator and s_t is the projected signal tensor. When Cpl
is the identity, this reduces to the uncoupled case. When Cpl is learned, it
can decorrelate overlapping projections, route information between oscillators,
and discover interaction structure that no single oscillator could find alone.

What changes across orders is not *how much* the state can hold — the 64 complex
degrees of freedom are the same — but *how the coupling is constrained*. Each
order imposes a different structural hypothesis on how oscillators interact. At
order 1, every pair of oscillators interacts independently (4096 free parameters
for 64 oscillators). At order 2, the oscillators are arranged on a grid and the
coupling decomposes into independent row and column parts (128 parameters). At
order 3, a cube with three independent coupling axes (48 parameters). The
savings compound: d² (order 1), p² + q² (order 2), p² + q² + r² (order 3),
while p × q × r = d stays constant. Each order is a bet that coupling
decomposes cleanly along independent axes; if the bet matches the data, the
constraint is the correct structure.

In the implemented Reciprocator the coupling is not a fixed low-rank
factorization but a state-conditioned, sequential composition of per-mode mixing
matrices. Each mode computes its mixing matrix from the partially-mixed tensor
after earlier modes have acted, making the operator fully data-dependent and
strictly more expressive than independent low-rank products. The original
low-rank intuition remains a useful mental model for why factorization is
powerful; the actual mechanism is richer.

## 9. Growing the Rank: Adaptation Through Parameter-Space Residuals

The engine does not commit to a fixed factorization. A single signal drives
everything: the **parameter-space residual** — the gap between what the signal
demands and what the current gain parameters can express.

Recall from §8 that the state is a tensor with R axes, each of size m_i. The
rank R is the number of axes. The degree — the total number of oscillators — is
the product m_1 × m_2 × ... × m_R. Growth and pruning operate on these two
levels differently: both mode sizes and rank can grow and shrink.

When a signal arrives, the gain parameters (decay, input, recurrent) along each
mode are reshaped into basis rows. The signal's mode-`m` marginal is projected
against the column space of these basis rows. The residual — what falls outside
that column space — measures the structural gap: the variation in the signal
that the current parameter space cannot express.

This residual drives two responses at different timescales.

**Continuously**, gradient descent flows back through the gain parameters and
coupling, adjusting the basis so that future readings are more accurate.

**Periodically**, the engine checks whether the current state has the right
shape. If a smoothed EMA of the per-mode residual norms exceeds a threshold,
the engine picks the mode with the highest residual and extends it by one
position — seeded from the orthogonal complement of the existing basis (the
direction the current parameters cannot represent). Extending a mode by one
position adds an entire slice of oscillators across all other modes, so the
degree grows by more than one. If all modes are structurally saturated (all
residuals below a saturation threshold) and the training loss remains high, the
engine adds a new axis entirely (matrix → 3-tensor → 4-tensor), increasing the
rank.

Mode sizes and rank can both shrink. The engine measures **cross-mode
redundancy**: for each mode, it computes the residual of that mode's embedded
signal against the union of all other modes' embedded signals. A mode whose
contribution is largely redundant with other modes is a candidate for pruning.
If the redundancy stays low for long enough (past a guard period since the last
growth event), the mode tail is removed by averaging, or the entire rank axis
is collapsed. Modes are never pruned below size 1.

Growth expands where the parameter-space residual is highest; pruning trims
where cross-mode redundancy is lowest. Together they allocate the engine's
representational budget toward whatever the data demands.

The code realizes both mode-size growth and dynamic rank increase, triggered by
SVD-based parameter-space residuals and pruned by cross-mode redundancy.

With adaptive capacity and self-relation in place, the same geometric operations
scale across physical organization — from the apple to the mind. The next section
traces that continuous cascade.

## 10. The Engine Inside the Language Model

A modern language model is a stack of identical blocks. Each block has two
paths: a *mixer* that combines information across positions, and a *feed-forward
network* (MLP) that transforms representations at each position independently.
In a Transformer, the mixer is attention. In Mamba, it is a selective state
space model. In our architecture, the mixer is the reciprocator engine (serial causal form).

### Attention

Attention is a retrieval operation. For each output position, it computes a
weighted sum over all input positions, with weights determined by query-key
compatibility. This gives it direct access to the full sequence — every token
can attend to every other token — but at quadratic cost in sequence length. The
state is implicit: the entire sequence is re-processed at every layer, every
step. There is no compressed summary carried forward.

### Mamba (Selective State Space)

Mamba replaces the full-sequence scan with a compact recurrent state updated at
each timestep. The state is a real-valued vector of fixed dimensionality. Input
is projected into the state through learned linear maps, the state evolves via a
discrete dynamics equation, and the output is projected back out. This gives
linear-time inference and a genuine compressed memory, but the coupling between
input and state is learned via unconstrained linear projections — no geometric
structure is imposed on how the state responds to signals.

### The Reciprocator Engine

The reciprocator sits in the same slot as attention or Mamba — it is the mixer
in each block. But it makes different commitments:

1. **The state is complex-valued.** Not a real vector, but a list of
   oscillators with magnitude and phase. This gives the engine a natural
   vocabulary for resonance (in-phase reinforcement), torque (quadrature
   rotation), and interference (anti-phase cancellation) — regimes that a
   real-valued state must learn implicitly.

2. **The state is factorized.** Not a flat vector, but a tensor whose rank and
   mode sizes can adapt during training. The coupling between input and state is
   not a single learned projection but a structured interaction with independent
   axes.

3. **The state persists.** It is carried across timesteps, not re-derived from
   the sequence at each layer. This is the same commitment as Mamba (and the
   opposite of attention), but the state is richer — a complex-valued
   factorized oscillator bank rather than a real-valued vector.

4. **The coupling is geometric.** The engine does not learn a generic linear
   map from input to state. It projects the input onto a learned basis,
   decorrelates via per-mode coupling matrices, and modulates through the
   Hadamard product with the current state. The state *filters* the input — it
   is not just transformed by it.

### Tokens and the MLP

A token enters the model as a learned complex number: phase from position
encoding, and magnitude from either a bounded learned embedding, an
inverse-frequency scaling applied to a normalized learned profile, or that same
inverse-frequency prior multiplied by a learned positive residual. By default
the token carries no inherent phase — it arrives as pure magnitude with phase
determined entirely by its position in the sequence. An optional token-phase
mechanism can add learned per-token phase offsets (semantic projections from the
token embedding, or virtual position offsets), but this is disabled by default
and zero-initialized so the baseline behavior is pure positional phase. This is
deliberate: the token's "direction" is primarily determined by its position, so
the engine's oscillators decide through their own phase structure what to do
with it.

Inside each block, the mixer reads the token through the state and produces a
delta — the change the token caused. The MLP does not receive the full
post-mixer state. It receives the delta: the difference between the state after
the token and the state before. This is the relational quantity. The delta is
not the token, and it is not the state — it is what happened when the token met
the state. It is reception (§1), compressed into a single vector. The MLP's job
is to interpret that reception: what does this change mean, and what should the
model do with it?

### The Full Block

Each block in the stack is:

    input → normalize → mixer (reciprocator) → normalize(Δ) → MLP → residual add → output

The mixer produces a delta — what changed when the token met the state. The MLP
receives this delta, not the full post-mixer hidden state. Its job is to
interpret the reception: what does this change mean, and what should the model
do with it? The MLP output is then added to the input as a single residual
connection.

The stack of blocks allows the model to build hierarchical representations. In
early layers, the reciprocator tracks local patterns (syntax, character
sequences). In deeper layers, it tracks abstract patterns (semantics, discourse
structure). The factorized tensor state at each layer is the engine's
compressed summary of what it has seen so far — a growing, structured memory.

## 11. Spectral Reciprocation: The Memory’s Multi-Scale Mirror

In addition to the sequential per-mode coupling (§8), the engine supports
spectral coupling operators that replace the sequential coupling with
frequency-domain alternatives. These let the memory look at itself through a
rich collection of “lenses” at many different scales at once — from the slowest,
broadest patterns (the deep bass of the symphony) all the way down to the
fastest, most detailed ripples.

**FFT spectral coupling** applies an N-dimensional FFT over the tensor axes,
multiplies by a smooth radial gain filter (boosting low frequencies, damping
high frequencies), and inverts back. The filter is isotropic and parameterized
by gain and width settings.

**DWT spectral coupling** flattens the tensor and applies a Haar wavelet
pyramid decomposition. Each detail band receives a gain derived from the same
spectral gain function, and the filtered coefficients are reconstructed by the
inverse Haar transform.

**Wavelet packet coupling** builds a full Haar wavelet packet tree over the
flattened tensor, then selects the single best basis — the view that
concentrates energy most cleanly while keeping the phases across the whole
memory in strong mutual agreement (a gauge-aware harmony). In the
entropy-only variant, the cost is energy entropy alone. In the gauge-aware
variant, cost = entropy + (1 − phase_coherence), where phase coherence
measures alignment with a global phase reference.

Once the best multi-scale view is chosen, the system gently strengthens the
coherent, meaningful parts (especially the slower, lower-frequency structures
that tend to carry long-term invariants) and softly quiets the scattered,
incoherent noise. The refined coefficients are reconstructed back into the
state, and the entire memory is returned to the unit hypersphere.

This turns hierarchical abstraction into a native geometric operation. The
memory is no longer just reacting token-by-token; it is listening to its own
internal structure at every timescale simultaneously, reinforcing the harmonies
that matter and letting transient noise fade. Long-range structure and
multi-level invariants emerge directly from the mathematics of reciprocation
itself.

## 12. Multi-Layer Coordination

Each layer contains a single tensor engine. The stack of layers creates
depth-based coordination: early layers track local patterns while deeper layers
track abstract patterns. The factorized tensor state at each layer is the
engine’s compressed summary of what it has seen so far.

The geometry stays clean and zero-sum. Nothing is added or removed from the
total “energy budget” of the hypersphere; the system simply rearranges its own
internal resonances so that the meaningful, coherent patterns grow stronger
together.

## Summary Equations (Core Engine)

The minimal update (§6):

    s_t      = normalize(W_mag, W_phi · input)       # lift signal into memory space
    Z        = s_t ⊙ S_{t-1}                         # relational product
    S_new    = Normalize(D ⊙ S_{t-1} + A ⊙ s_t + B ⊙ Z)  # update + compress

The equations above describe the minimal uncoupled case. The full engine (used
in the Reciprocator) augments this with sequential per-mode coupling (§8) that
replaces the diagonal `Z` with a state-conditioned mixing `R = Cpl(Z)`,
optional self-relation (§7), and a gated return map (reciprocator.md §9).
Normalization may be Frobenius or per-mode iterative. See reciprocator.md §6
for the complete update rule.

The full gated update with coupling (§8):

    s_t      = normalize(W_mag, W_phi · input)            # project signal (polar form)
    Z        = s_t ⊙ S_{t-1}                              # relational product
    R        = Cpl(Z)                                     # coupled reception (per-mode mixing)
    S̃        = D ⊙ S_{t-1} + A ⊙ s_t + B ⊙ R             # entrywise gains
    S_new    = Normalize(S̃)                                # compression

    where D = sigmoid(decay_logit),     in (0, 1)
          A = sigmoid(input_logit),     in (0, 1)
          B = tanh(recurrent_logit),    in (-1, 1)

Optional self-relation (§7):

    S̃_tent  = Normalize(S̃)
    S_new   = Normalize(S̃ + Λ_self · (S_{t-1} ⊙ S̃_tent))

Dynamic growth (§9):

    if ema_residual_m > threshold:
      M_m ← M_m + 1                                     # extend mode
      new_slice ← orthogonal_complement(basis_rows, signal_residual)
      S' ← Normalize(append(S', new_slice))

The apple falls. The mind wonders why. Both are made of the same physics. Only
one has learned to push back on itself, arranging its own oscillators to
resonate with the structure of the world.



## Prior Art

The engine sits at the intersection of three lines of work: complex-valued
representation (HRR, hyperdimensional computing), recurrent state models with
compressed memory (Mamba, SSMs), and retrieval-as-attention (Hopfield Networks,
Transformers). The novel combination is a complex-valued, factorized, persistent
state with geometrically structured coupling that adapts online. This section
traces each thread so future work can differentiate rather than re-motivate from
first principles.

### Complex-Valued Prototype Composition and Binding

- Plate, T. A. (1995). *Holographic Reduced Representations*. IEEE Transactions
  on Neural Networks 6(3), 623-641. Circular-convolution binding of unit-norm
  vectors. The engine shares HRR's composition primitive — binding via phase
  rotation — but uses it for recurrent state dynamics, not static
  representation. HRR composes; the engine evolves.
- Kanerva, P. (2009). *Hyperdimensional Computing*. Cognitive Computation 1(2),
  139-159. Vector symbolic architectures over high-dimensional unit-norm
  carriers. The engine's unit-norm constraint is the same; the difference is
  that hyperdimensional computing stores and retrieves from a fixed codebook,
  while the engine's basis grows online from parameter-space residuals.

### Unit-Norm and Complex-Valued Recurrent Dynamics

- Arjovsky, M., Shah, A., Bengio, Y. (2016). *Unitary Evolution Recurrent Neural
  Networks*. ICML. Recurrence constrained to the unit-modulus manifold for
  stable long-range propagation. The engine shares the stability motivation but
  uses unit-norm (magnitude + phase) rather than strict unit-modulus (phase
  only), allowing the state to express confidence alongside direction.
- Trabelsi, C. et al. (2018). *Deep Complex Networks*. ICLR. Complex-valued
  layers with magnitude/phase normalization. Established that complex-valued
  representations are learnable at scale; the engine builds on this by making
  the phase structure operationally meaningful (resonance, torque, interference)
  rather than an implicit learned property.

### Retrieval Dynamics Over a Fixed Dictionary

- Ramsauer, H. et al. (2020). *Hopfield Networks is All You Need*. ICLR 2021.
  Modern continuous Hopfield retrieval from stored patterns. The engine's
  coupled-reception readout has similar retrieval structure, but the state
  modulates the retrieval via phase-sensitive Hadamard product rather than
  energy minimization, and the coupling is data-dependent rather than fixed.

### Prediction-Error Dynamics

- Rao, R. P. N., Ballard, D. H. (1999). *Predictive coding in the visual cortex*.
  Nature Neuroscience 2, 79-87. Prediction-error minimization as recurrent
  cortical computation. The engine's model-level anticipator (reciprocator.md
  §11) uses a related prediction-feedback loop — converting output predictions
  into hidden-state anticipations — while the growth mechanism uses
  parameter-space residuals rather than prediction error directly.

### Online Basis Growth by Residual

- Mairal, J., Bach, F., Ponce, J., Sapiro, G. (2009). *Online Dictionary
  Learning for Sparse Coding*. ICML. Streaming dictionary updates via residual
  reconstruction.
- Aharon, M., Elad, M., Bruckstein, A. (2006). *K-SVD*. IEEE Transactions on
  Signal Processing 54(11), 4311-4322. Residual-driven atom replacement and
  growth. Both establish the principle of online basis expansion from residuals;
  the engine applies a related principle inside a recurrent state model, where
  the residual is measured in parameter space (gain-field expressiveness) rather
  than data space, and new mode slices seed new state dimensions.

### Attention as Soft Retrieval

- Bahdanau, D., Cho, K., Bengio, Y. (2015). *Neural Machine Translation by
  Jointly Learning to Align and Translate*. ICLR 2015. Attention as learned
  soft-alignment over a key-value store.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
  Multi-head self-attention over sequences. The engine's coupled-reception
  readout shares the retrieval structure of attention — project queries against
  keys, weight by compatibility, aggregate values — but the coupling is
  data-dependent (computed from the signal-state product, not from a fixed key
  store), and the state modulates the result via the phase-sensitive Hadamard
  product. Two structural differences: the state is persistent (carried across
  timesteps, not re-derived from the sequence), and the coupling is a
  sequential composition of per-mode mixing matrices rather than a flat
  query-key-value projection.

### Selective State Spaces and Gated Recurrence

- Gu, A., Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces*. arXiv:2312.00752. Continuous-time state-space models with
  input-dependent selection, achieving linear-time inference. The engine shares
  Mamba's core commitment — a compact recurrent state updated at each timestep
  rather than materialized over the full sequence — but differs in three ways:
  (1) the state is complex-valued and unit-norm, not real-valued; (2) the
  coupling between input and state is geometrically structured (phase-aware,
  with explicit resonance/torque/interference regimes) rather than learned via
  linear projections; (3) the state is a factorized tensor whose rank and mode
  sizes adapt during training, whereas SSMs fix the state dimensionality.

No prior work combines a complex-valued persistent state with factorized
adaptive-rank coupling and online capacity growth driven by parameter-space
residuals. Each component has precedent; the contribution is their integration
into a single recurrent mixer with geometrically interpretable dynamics.

---

**Implementation Note.** This essay presents the geometric intuition. The
precise mathematical specification, full update equations, training procedure,
and hardware-efficient kernels live in reciprocator.md and the code in model.py.
