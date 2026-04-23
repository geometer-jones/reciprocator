# Reciprocator: Architecture and Mathematics

This document specifies the Reciprocator as a causal complex tensor state-space
architecture. Its central object is a rank-`r` complex tensor memory that
evolves online while interacting with a complex hidden stream. Where the
mathematical form is an abstraction whose instantiation is numerical rather
than closed-form, that is stated explicitly.

## 1. Core Objects

Given a token sequence `x_{1:T}`, the model evolves:

- a complex hidden stream
  `h_t^{(\ell)} in C^D`
- a complex tensor memory state
  `S_t^{(\ell)} in C^{M_1 x ... x M_r}`

Indices:

- `t = 1, ..., T` is token position
- `\ell = 1, ..., L` is layer index
- `r` is tensor rank (number of modes)
- `(M_1, ..., M_r)` are mode sizes, with `M_state = prod_m M_m`

The model is causal: for every layer `\ell` and time `t`, `h_t^{(\ell)}` depends
only on `x_{1:t}`.

Each layer contains a single tensor mixer (no engine bank). The mixer reads the
hidden stream through its tensor state and produces a delta correction to the
hidden stream.

## 2. Token Lifting

Tokens enter the model as an amplitude modulation of an all-ones carrier;
positions enter as a rotary phase. Three token-magnitude parameterizations are
supported for token `x_t` at position `t`:

```math
r^{learned}_{tok}(x_t)[d] = exp( 0.1 * tanh( a_tok(x_t)[d] ) )          (real, in (e^{-0.1}, e^{0.1}))

p_tok(x_t)[d] = softplus( a_tok(x_t)[d] )
hat{p}_tok(x_t) = p_tok(x_t) / ||p_tok(x_t)||_2
s_tok(x_t) = bar{f} / f(x_t)
r^{invfreq}_{tok}(x_t)[d] = s_tok(x_t) * hat{p}_tok(x_t)[d]
delta_tok(x_t) = exp( b_tok(x_t) )
r^{invfreq+learn}_{tok}(x_t)[d] = delta_tok(x_t) * r^{invfreq}_{tok}(x_t)[d]
```

where `a_tok` is a learned embedding and `f(x_t)` is the token frequency
(defaulting to a Zipf prior by token rank when no explicit frequency table is
supplied). In `learned` mode, `rho_tok = r^{learned}_{tok}`. In
`inverse_frequency` mode, `rho_tok = r^{invfreq}_{tok}`: the embedding learns
only the within-token profile while overall token norm is set by inverse
frequency. In `inverse_frequency_learned` mode, `rho_tok = r^{invfreq+learn}_{tok}`:
the model starts from the same inverse-frequency prior but multiplies it by a
learned positive scalar residual initialized to 1.

### Positional phase

Three positional phase types are supported:

**RoPE.** Standard rotary position encoding with frequencies
`omega_d = base^{-d/D}` (`base = 10000`):

```math
rho_pos(t)[d] = exp( i * t * omega_d )
```

**Locked wave.** A single learned frequency and wavevector shared across all
dimensions:

```math
omega = softplus(log_omega)
rho_pos(t)[d] = exp( i * (omega * t + log_wavevector * d) )
```

**Local wave.** Per-band frequencies and wavevectors. The hidden dimension is
partitioned into `ceil(sqrt(D))` contiguous bands, each with its own learned
frequency and wavevector:

```math
omega_b = softplus(log_omega_bands[b(d)])
rho_pos(t)[d] = exp( i * (omega_b * t + log_wavevector_bands[b(d)] * d_in_band(d)) )
```

where `b(d)` is the band assignment for dimension `d` and `d_in_band(d)` is the
within-band offset.

### Token phase

An optional per-token phase offset adds semantic or positional phase variation:

- `none`: no token phase; the carrier depends only on position.
- `semantic`: a learned linear projection of the token embedding adds a
  per-token phase offset (zero-initialized, so inert at start):
  `phi_tok = W_phi_proj * a_tok(x_t)`.
- `virtual_offset`: a learned per-token scalar `alpha(x_t)` shifts the position
  phase: `phi_tok = alpha(x_t) * omega` (RoPE) or
  `phi_tok = alpha(x_t) * omega_b[d]` (local_wave).
- `semantic_virtual_offset`: both semantic and virtual-offset contributions.

### Lifting

The lifted representation combines token magnitude and positional phase:

```math
h_t^{(0)} = rho_tok(x_t) odot rho_pos(t)
```

The token contribution is purely amplitude; the position contribution is purely
phase. An optional per-token phase makes the carrier token-dependent.

### Distribution lifting

A soft distribution over tokens can be lifted by computing the weighted sum of
per-token lifted representations:

```math
lift(p) = sum_x p(x) * (rho_tok(x) odot rho_pos(t+1))
```

This is used by the anticipator relation (§11).

## 3. Layer Architecture

Each layer is a residual block where the mixer produces a delta that is
transformed by the FFN before being added:

```math
u_t^{(\ell)} = cLN( h_t^{(\ell-1)} )
Delta_t^{(\ell)}, S_t^{(\ell)} = M^{(\ell)}( u_t^{(\ell)}, S_{t-1}^{(\ell)} )
h_t^{(\ell)}   = h_t^{(\ell-1)} + cFFN^{(\ell)}( cLN( Delta_t^{(\ell)} ) )
```

`cLN` is complex layer normalization (mean-subtract then rescale by magnitude
variance). `cFFN` is a complex feedforward network using a modReLU nonlinearity
that thresholds magnitude while preserving phase. The FFN receives the mixer's
delta output, not the full post-mixer hidden state. The FFN therefore interprets
what changed when the token met the state, not the state itself.

`M^{(\ell)}` is the mechanism that transports information across positions into
the tensor state in the Reciprocator-only architecture. Optional cross-layer
state injection can then read out that span summary into the next layer's
hidden stream, but it does not create a second tensor-state update path.

### Optional cross-layer state injection

An optional read-only path (`enable_cross_layer_state=True`) lets layer `\ell`
donate a correction derived from its tensor state to the hidden stream consumed
by layer `\ell+1`. The layer states themselves remain disjoint; the only effect
is an additive correction to hidden.

Let `H_tau^{(\ell)}` be the hidden span produced by layer `\ell` on the current
forward span `tau` (the whole sequence in the exact path, or one chunk in the
chunked path), and let `S_{tau,end}^{(\ell)}` be the final tensor state of that
same span. Reusing the phase-aware state feature map `psi(S)` from §9:

```math
q_cross^{(\ell)} = W_cross^{(\ell)} * psi( S_{tau,end}^{(\ell)} )   in R^D
g_cross^{(\ell)} = tanh( beta_cross^{(\ell)} )                      in (-1, 1)

H_tau^{(\ell)} <- H_tau^{(\ell)} + g_cross^{(\ell)} * complex( q_cross^{(\ell)}, 0 )
```

The injected correction is therefore real-valued before complex casting, and it
is broadcast across the entire recipient span. The gate `beta_cross^{(\ell)}`
is zero-initialized, so the path is exactly inert at initialization even when
enabled.

This is a read-only inter-layer memory path, not a state-to-state write. Layer
`\ell+1` can condition on a learned summary of `S_{tau,end}^{(\ell)}`, but it
does not mutate or merge the lower layer's tensor state.

## 4. Tensor Signal

The normalized hidden state is lifted into a complex tensor signal via a polar
parameterization using two real linear maps over the flattened complex hidden
features `[Re(u_t), Im(u_t)]`:

```math
m_t^{(\ell)}   = softplus( W_mag^{(\ell)} * [Re(u_t), Im(u_t)] )       in R^{M_state}_{>0}
phi_t^{(\ell)} = pi_phi * tanh( W_phi^{(\ell)} * [Re(u_t), Im(u_t)] )   in (-pi_phi, pi_phi)^{M_state}
Y_t^{(\ell)}   = m_t^{(\ell)} odot exp( i * phi_t^{(\ell)} )            in C^{M_state}
X_t^{(\ell)}   = reshape( Y_t^{(\ell)}, (M_1, ..., M_r) )               in C^{M_1 x ... x M_r}
s_t^{(\ell)}   = N( X_t^{(\ell)} )
```

`W_mag` and `W_phi` are real linear maps. The softplus constraint enforces
strictly positive amplitude. The `pi_phi * tanh` constraint confines the
per-feature phase; with `pi_phi = pi` (the default), the full angular range is
covered without wrap-around discontinuity. The polar form keeps amplitude and
phase as structurally separate learning targets with explicit geometric meaning.

The learned `W_mag` and `W_phi` together absorb the role that a
Tikhonov-regularized Gram inverse `(E^* E + epsilon I)^{-1}` would play in a
fixed-dictionary formulation; there is no explicit Gram correction in the
architecture (see §16).

## 5. Normalization

Two normalization families are supported.

### Frobenius normalization

```math
N_F(S) = S / ||S||_F
```

### Per-mode normalization

Iterative alternating mode projections:

```math
N_PM(S) = (P_r circ ... circ P_1)^n (S)
```

where `P_m` rescales each nonzero mode-`m` fiber by its `l_2` norm. The
iteration is unrolled to a fixed depth (default 8 sweeps).

## 6. Mixer Dynamics

Each layer contains a single mixer that maintains one tensor state
`S_t^{(\ell)}`. The mixer operates in three phases: signal projection, coupling,
and gated update.

### Base gain fields

Per-layer static parameter tensors of shape `(M_1, ..., M_r)` under bounded
parameterizations:

```math
D^{(\ell)} = sigmoid( D_raw^{(\ell)} )    # decay,      in (0, 1)
A^{(\ell)} = sigmoid( A_raw^{(\ell)} )    # input gain, in (0, 1)
B^{(\ell)} = tanh   ( B_raw^{(\ell)} )    # recurrent,  in (-1, 1)
```

### Relational dynamic gain modulation

An optional relational projector (`dynamic_gains=True`) turns the three gain
fields into functions of the current lifted signal `s_t^{(\ell)}` while
preserving the same bounded update form. The projector reads the signal in the
state's own geometric coordinate system: flatten the complex tensor, split into
real and imaginary parts, pass through a tiny low-rank MLP, then reshape back
to three state-shaped delta fields.

For bottleneck rank `r = gain_projector_rank`:

```math
g_t^{(\ell)} = [Re(vec(s_t^{(\ell)})), Im(vec(s_t^{(\ell)}))]      in R^{2 M_state}

Delta_t^{(\ell)} = W_2^{(\ell)} * ReLU( W_1^{(\ell)} * g_t^{(\ell)} )
                  in R^{3 M_state}

[delta_D, delta_A, delta_B] = reshape( Delta_t^{(\ell)}, (3, M_1, ..., M_r) )
```

The modulated logits are then

```math
D_{logit,t}^{(\ell)} = D_raw^{(\ell)} + alpha_D^{(\ell)} * delta_D
A_{logit,t}^{(\ell)} = A_raw^{(\ell)} + alpha_A^{(\ell)} * delta_A
B_{logit,t}^{(\ell)} = B_raw^{(\ell)} + alpha_B^{(\ell)} * tanh(delta_B)
```

with bounded gains

```math
D_t^{(\ell)} = sigmoid( D_{logit,t}^{(\ell)} )
A_t^{(\ell)} = sigmoid( A_{logit,t}^{(\ell)} )
B_t^{(\ell)} = tanh( B_{logit,t}^{(\ell)} )
```

This is relational because the projector depends on the current signal tensor,
rotational because it operates on the signal after polar lifting into state
coordinates, and constructive because the modulation is generated by one ordered
low-rank procedure inserted directly into the existing update path.

The path is zero-initialized twice: the projector's final linear layer starts at
zero and the scalar mixing coefficients `alpha_D`, `alpha_A`, `alpha_B` start
at zero. So when `dynamic_gains=False`, or at initialization when it is
enabled, the mixer is exactly the original static-gain system.

The feature is exposed as `dynamic_gains` and `gain_projector_rank` on
`TrainingConfig`, `ReciprocatorLM`, and `ReciprocatorBlock`.

### Learned initial state prior

The tensor state no longer cold-starts from the origin. Each mixer owns a
trainable complex prior `S_0^{(\ell)}` with the same shape as the tensor state.
At parameter initialization it is sampled at small magnitude and projected onto
the chosen normalization geometry:

```math
\Xi_0^{(\ell)} ~ 0.01 * \mathcal{N}_C(0, I)
S_0^{(\ell)} = N( \Xi_0^{(\ell)} )
```

At runtime this learned prior is broadcast across the batch and used wherever a
fresh state is required (sequence start, stream wrap, or model rebuild after
growth/pruning). The recurrent dynamics for `t >= 1` are unchanged; the only
difference is that the first relational product uses a learned normalized seed
instead of the zero tensor:

```math
Z_1^{(\ell)} = s_1^{(\ell)} odot S_0^{(\ell)}
```

The prior is normalized once at initialization, not re-projected on every call
to `initial_state`. After initialization it is trained like any other mixer
parameter.

### Relational product and coupling

The signal meets the previous state through the elementwise Hadamard product,
then the coupling operator redistributes energy across modes:

```math
Z_t^{(\ell)} = s_t^{(\ell)} odot S_{t-1}^{(\ell)}
R_t^{(\ell)} = Cpl^{(\ell)}( Z_t^{(\ell)} )
```

`Cpl` is the only non-diagonal operator in the update. Its form depends on the
coupling type (§8).

### Update

The full update combines decay, input, and coupled-recurrent terms:

```math
widetilde{S}_t^{base,(ell)}
  = D_t^{(\ell)} odot S_{t-1}^{(\ell)}
  + A_t^{(\ell)} odot s_t^{(\ell)}
  + B_t^{(\ell)} odot R_t^{(\ell)}

S_t^{(\ell)} = N( widetilde{S}_t^{base,(ell)} )
```

All gains are entrywise and bounded. In the static case,
`D_t = D, A_t = A, B_t = B`; with dynamic gains enabled, they become
token-conditional via the relational projector above. Decay constrains how much
previous state persists; input gain controls how strongly the new signal
enters; the recurrent gain scales the coupling output.

## 7. Self-Relation

An optional self-relation mechanism (enabled by `enable_self_relation`) adds a
second relational product: the previous state meets the tentative updated state
before final normalization.

```math
bar{S}_t^{(\ell)} = N( widetilde{S}_t^{base,(ell)} )
Z_t^{self,(ell)} = S_{t-1}^{(\ell)} odot bar{S}_t^{(\ell)}

widetilde{S}_t^{(\ell)}
  = widetilde{S}_t^{base,(ell)}
  + Lambda_self^{(\ell)} odot Z_t^{self,(ell)}

S_t^{(\ell)} = N( widetilde{S}_t^{(\ell)} )
```

`Lambda_self = tanh(self_relation_logit)` is parameterized as a gain field of
shape `(M_1, ..., M_r)` and initialized to zero, so the path is inert at
initialization and activates only if the optimizer finds it beneficial.

Without self-relation, the state is purely reactive: it responds to each signal
and then waits passively for the next one. Self-relation gives the state a
second source of pressure from its own recent history. Where the state has been
consistent with its trajectory, resonance reinforces that direction; where it
has shifted, torque or interference correct.

## 8. Mode Coupling

`Cpl^{(\ell)}` is the coupling operator that redistributes energy across tensor
modes. Five coupling types are supported.

### Sequential mode coupling

A sequential composition of complex per-mode mixing matrices. Fix a mode `m`.
Let `Z^{(m-1)}` be the tensor after modes `1, ..., m-1` have been applied (with
`Z^{(0)} = Z_t^{(\ell)}`). Flatten so that mode `m` indexes the rows:

```math
Score_m = ( W_m * Z_m ) * Z_m^T                             in C^{M_m x M_m}
```

The coupling matrix has magnitude-row-stochastic routing and unit-phasor
direction:

```math
T_m = softmax( |Score_m| / (tau * sqrt(M_state / M_m)) ) odot ( Score_m / |Score_m| )
```

where `tau > 0` is a learned coupling temperature. The softmax selects
destination weights by magnitude; the unit-phasor preserves the directional
rotation from the bilinear score. Modes are applied sequentially, each computed
from the partially-mixed tensor after earlier modes:

```math
Z^{(0)} = Z_t^{(\ell)}
Z^{(m)} = T_m *_m Z^{(m-1)},  m = 1, ..., r
R_t^{(\ell)} = Z^{(r)}
```

Because `T_m` is computed from `Z^{(m-1)}` (not `Z^{(0)}`), later-mode routing
adapts to what earlier modes have already mixed.

### FFT spectral coupling

An N-dimensional FFT over all tensor axes followed by a smooth radial spectral
gain filter and inverse FFT:

```math
F = FFT_n(S)
R = IFFT_n( F * G(|k| / max|k|) )
```

The spectral gain `G(nu)` boosts low frequencies and damps high frequencies:

```math
G(nu) = (1 + g_low * exp(-nu^2 / (2 sigma^2)) - g_high * sigma((nu - c_high) / w))_+
```

where `g_low` is the low-frequency gain, `sigma` is the low-frequency width,
`g_high` is the high-frequency gain, `c_high` is the high-frequency cutoff, and
`w` is a transition width derived from the cutoff as
`w = max(0.05, 0.25 * max(0.05, 1 - c_high))` (widening the sigmoid when the
cutoff is low, clamped to a minimum of 0.05). The filter operates on radial
frequency only (isotropic in the N-dimensional frequency domain).

### Dynamic spectral gain modulation

Spectral backends can replace the fixed-only envelope with a learned
signal-conditioned envelope by setting `dynamic_spectral_gains=True`. The fixed
gain `G(nu)` remains the base envelope and is still controlled by
`low_frequency_gain`, `low_frequency_sigma`, `high_frequency_gain`, and
`high_frequency_cutoff`.

For the current coupling signal `Z_t`, flatten the complex state, split it into
real and imaginary coordinates, and pass it through the same low-rank projector
shape used by dynamic mixer gains:

```math
h_t = [Re(vec(Z_t)), Im(vec(Z_t))]                         in R^{2 M_state}
delta g_t = W_2 * ReLU(W_1 * h_t)                          in R^{M_state}
```

For FFT coupling, `delta g_t` is interpreted as learned samples of a radial
one-dimensional envelope by default. The samples are interpolated at each
coordinate's normalized radius and added to the radial base map per batch item:

```math
G_t = (G + alpha_spectral * radial(delta g_t, nu))_+
R_t = IFFT_n( FFT_n(Z_t) * G_t )
```

Setting `anisotropic_spectral_gains=True` switches FFT dynamic modulation from
radial samples to a full coordinatewise frequency-grid map:

```math
G_t = (G + alpha_spectral * reshape(delta g_t))_+
R_t = IFFT_n( FFT_n(Z_t) * G_t )
```

For DWT and wavelet-packet coupling, the same projected `delta g_t` is averaged
over the flattened frequency interval for each Haar band or selected packet
leaf, then added to that band's fixed gain. This preserves the existing banded
spectral structure while making the band gains depend on the current signal.

The dynamic path is inert at initialization: the projector's final linear layer
is zero-initialized and `alpha_spectral` starts at zero. With
`dynamic_spectral_gains=False`, or at initialization when it is enabled, all
spectral backends therefore match the fixed-filter behavior exactly. The
anisotropic flag is also inert unless `dynamic_spectral_gains=True`; static
`fft` and static `fft + anisotropic_spectral_gains` should produce the same
result for the same seed and fixed envelope.

The planned FFT gain ablation isolates these effects with five arms:

| coupling_type | dynamic_spectral_gains | anisotropic_spectral_gains | Purpose |
|---------------|------------------------|----------------------------|---------|
| `sequential`  | `False`                | `False`                    | Direct recurrent baseline |
| `fft`         | `False`                | `False`                    | Fixed radial FFT baseline |
| `fft`         | `True`                 | `False`                    | Learned radial FFT gains |
| `fft`         | `False`                | `True`                     | Inert anisotropic-flag control |
| `fft`         | `True`                 | `True`                     | Learned full coordinatewise anisotropic FFT gains |

This comparison separates three questions: whether FFT helps at all, whether
the learned radial projector helps beyond the fixed envelope, and whether
coordinatewise anisotropy helps beyond learned radial modulation.

### DWT spectral coupling

The tensor is flattened to 1D and decomposed via a Haar wavelet pyramid. Each
detail band receives a per-band gain derived from the spectral gain function.
The approximation coefficients at the deepest level also receive a gain. The
filtered coefficients are reconstructed by the inverse Haar transform.

### Wavelet packet coupling

A full Haar wavelet packet tree is built over the flattened tensor. The best
basis is selected by minimizing a per-node cost function over the tree:

- `wavelet_packet`: cost = energy entropy `H = -sum(p log p)` where `p_i =
  |c_i|^2 / sum|c|^2`.
- `wavelet_packet_max_gauge`: cost = energy entropy + `(1 - phase_coherence)`,
  where phase coherence measures how well the node's coefficients align with a
  global phase reference (the mean unit-phasor of the flattened tensor).

The recursive selection compares the cost of keeping a node as a leaf against
the cost of splitting it into its approximation and detail children. Once the
best basis is chosen, each leaf node is multiplied by its band gain and the tree
is reconstructed.

## 9. Hidden-Space Return Map

The mixer summarizes the updated state into a phase-aware real feature map and
maps it back to the hidden dimension through a gated return. Let
`bar{S}_t^{(\ell)} = (1 / M_state) sum_j S_t^{(\ell)}[j]` be the mean complex
state value and define the gauge-invariant cross-product
`c_t^{(\ell)} = S_t^{(\ell)} odot overline{bar{S}_t^{(\ell)}}`:

```math
psi(S) = [ Re(c), Im(c), |S| ]                             in R^{3 M_state}
d_t^{(\ell)} = W_ret^{(\ell)} * psi( S_t^{(\ell)} )          in C^D
g_t^{(\ell)} = sigmoid(
                    W_gate^{(\ell)} *
                    [ Re(u_t^{(\ell)}), Im(u_t^{(\ell)}),
                      Re(d_t^{(\ell)}), Im(d_t^{(\ell)}) ]
                  )                                            in R^D
Delta_t^{(\ell)} = g_t^{(\ell)} odot d_t^{(\ell)}
```

The gate is real and broadcast elementwise over both real and imaginary
components of `d_t`. `W_ret` maps real features to complex outputs via a
`RealToComplexLinear` map. This uses a data-dependent U(1)-invariant feature
map (cf. the learned-anchor readout in §10, which trades gauge-invariance for
a trainable phase alignment).

## 10. Readout Geometry

After the final layer the hidden state remains complex. Two real-valued
readouts are supported.

### Magnitude readout

```math
R_mag(h_t) = | h_t |                                              in R^D
```

### Phase-aware readout (learned-anchor bilinear)

A learned complex anchor vector `a in C^D` (initialized to `1+0i`, normalized to
unit magnitude at each forward pass) serves as the phase reference:

```math
reference = a / |a|                                                in C^D
c_t = h_t odot overline{reference}                                 in C^D
R_phase(h_t) = [ Re( c_t ), Im( c_t ), | h_t | ]                  in R^{3D}
```

Unlike the return map (§9), which uses the data-dependent feature mean for
strict U(1)-invariance, the readout uses a learned reference. This trades
gauge-invariance for the ability to learn a task-specific phase alignment
between the hidden state and the output projection. The anchor is a trainable
parameter of shape `(D,)`, initialized to `1+0i` everywhere.

Vocabulary logits are produced by a real output map:

```math
z_t = W_out * R( h_t^{(L)} )
```

## 11. Anticipator Relation

An optional anticipator mechanism (enabled by `enable_anticipator_relation`)
creates a feedback loop where a lifted next-token signal modulates the current
input before the block stack. The modulation is applied once per forward pass,
before iterating over the blocks:

```math
h_t' = h_t + Lambda_anticip odot ( h_t odot a_t )
```

where `Lambda_anticip = tanh(anticipator_relation_logit)` is a learned
per-dimension gain initialized to zero. The anticipator is therefore inert at
initialization and only activates if training finds it useful.

The anticipatory signal `a_t` has three cases.

### Teacher-forced oracle path

During training (or any forward call that supplies `targets`), the next-token
ids are lifted directly:

```math
a_t = lift( x_{t+1} )
```

implemented as a one-position shift of `targets` via `TokenLift`. The final
sequence position has no in-chunk next token, so its oracle anticipator is
masked out:

```math
a_T = 0
```

This prevents a dummy chunk-boundary token from leaking a training signal into
the modulation term.

### Inference carry path

At inference time, if a carried anticipatory hidden state is supplied from the
previous forward call, that carried state is reused:

```math
a_t = hat{h}_t
```

If no carried anticipatory state is available (cold start), the anticipator is
zero:

```math
a_t = 0
```

### Next-step prediction for the next call

When `targets` are not supplied, the model also produces a next anticipatory
hidden state from its own output logits. Rather than lifting the full softmax,
the implementation keeps the top 8 vocabulary entries per position, renormalizes
them, and lifts that sparse distribution:

```math
tilde{p}_{t+1} = renorm( top8( softmax(z_t) ) )
hat{h}_{t+1} = lift( tilde{p}_{t+1} )
```

This `hat{h}_{t+1}` is returned as `next_ant` so generation code can thread it
into the next call as `anticipatory_hidden`. Predicting the anticipatory hidden
state from output probabilities rather than from internal hidden state keeps the
mechanism tied to the model's explicit token predictions instead of its private
internal update path.

## 12. Capacity Allocation

The tensor state is defined over a shape `(M_1, ..., M_r)` that can change
during training through mode-size growth, rank growth, and pruning.

### Growth trigger: SVD-based residual norms

Growth decisions are based on per-mode residual norms computed via SVD. For each
mode `m`, the current tensor signal (averaged over batch and sequence) is
collapsed along mode `m` to form a candidate vector. The existing gain fields
(`decay_logit`, `input_logit`, `recurrent_logit`, and optionally
`self_relation_logit`) are reshaped along mode `m` into basis rows. The
residual norm is the component of the candidate that falls outside the column
space of the existing basis:

```math
residual_m = || candidate_m - Q_m (Q_m^H candidate_m) ||_2
```

where `Q_m` is the orthonormal basis from the SVD of the existing gain-logit
rows along mode `m`. A large residual indicates that the current gain-field
structure cannot express the signal's variation along that mode — there is
novel structure that the existing dimensions cannot capture.

An EMA smooths these residual norms over growth checks:

```math
ema_m = alpha * residual_m + (1 - alpha) * ema_m
```

### Mode-size growth

When `dynamic_mode_growth` is enabled and the EMA residual for some mode
exceeds `growth_residual_threshold`, the mode with the highest EMA residual is
extended by one position (incrementing `M_m`). The new slice is initialized by
one of four strategies:

- `zero`: the new slice is filled with zeros.
- `mean`: the new slice is filled with the mean of existing slices along that
  mode.
- `orthogonal`: the new slice is initialized to the QR orthogonal complement of
  the signal direction that triggered growth, targeting a direction not already
  spanned by the existing basis.
- `residual`: the new slice is initialized from the raw trigger direction
  itself, rescaled to match the typical norm of existing slices. Unlike
  `orthogonal`, this does not project out overlap with the existing basis.

### Rank growth

When `dynamic_rank_growth` is enabled, all mode residuals are below
`residual_saturate_threshold` (all modes are structurally saturated), and the
average training loss exceeds `rank_growth_loss_ceiling`, a new axis of size 2
is appended to the state shape. The new mode's parameters are initialized using
`rank_init`:

- `zero`: the new rank slice is filled with zeros.
- `mean`: the new rank slice is filled with the mean state over existing modes.
- `residual`: the new rank slice is seeded from the hidden/state residual
  accumulated at the trigger point and then norm-matched to the existing state.

### State dict transfer

After a growth event, the model is rebuilt with the new state shape and the old
parameters are transferred via shape-aware pad-copy:
- Existing slices in all state-shaped parameters (gain fields, signal
  projector weights/biases, return map weights) are preserved in place.
- The new slice receives the chosen initialization.
- Coupling mode-weight matrices for the grown mode are padded with one new row
  and column.
- Mode-weight matrices for other modes are copied unchanged.

## 13. Dynamic Pruning

Pruning removes underused dimensions by collapsing a mode or rank axis.

### Pruning trigger: cross-mode redundancy

Pruning decisions use a per-mode redundancy measure. For each mode `m`, each
slice along mode `m` is embedded into the full state space, and its residual
against the union of all other modes' embedded slices is computed. A low
residual means the slice adds negligible new signal beyond what other modes
already capture.

An EMA smooths these redundancy norms. A mode or rank axis becomes pruning-
eligible only when all of the following hold:

- its smoothed redundancy is below `prune_threshold`
- it has stayed below that threshold for `prune_sustain_steps` consecutive
  growth/prune checks
- more than `prune_min_steps` have elapsed since its last growth event

Among eligible modes, the mode with the lowest smoothed redundancy is selected
for mode-size pruning. Among eligible rank axes, the axis with the lowest
smoothed redundancy is selected for rank pruning.

### Mode pruning

Mode pruning reduces the size of the identified mode by one. The specific slice
removed is not chosen by position; it is chosen by a second EMA that tracks
per-slice activation variance. Within the already-selected mode, the slice with
the lowest activation-variance EMA is removed, on the assumption that it carries
the least informative direction. State-shaped parameters and the coupling weight
matrix for that mode are shrunk by deleting that slice's row/column rather than
dropping the tail or averaging the mode.

### Rank pruning

Removes an entire rank axis (collapsing the tensor by one mode). All parameters
are remapped accordingly.

Growth and pruning currently require `coupling_type="sequential"`.

## 14. Streaming Form

The serial mixer only requires:

- the current hidden state
- the previous tensor state

so the model admits a true streaming regime. Once a prefix has built the state
`{ S_t^{(\ell)} }_{\ell}`, advancing by one new token does not require
replaying the consumed prefix.

### Two timescales of learning

The architecture supports two distinct learning processes:

- **Reciprocation learning** (fast): in-context adaptation via state update.
  At each token, `S_t` accumulates relational, rotational structure from the
  input stream. This is bounded by state capacity, resets between sequences,
  and does not generalize across contexts. It is operative during both
  inference and training.

- **Gradient learning** (slow): parameter update via gradient descent. Tunes
  `W_mag`, `W_phi`, `W_ret`, `W_gate`, gain fields, and coupling parameters
  so that reciprocation learning tracks signal rather than noise. Persists
  across contexts. The quality of reciprocation learning is gated by what
  gradient learning has shaped the parameters to do.

The two are not symmetric: gradient learning determines what reciprocation
learning can usefully accumulate; without it, the state dynamics are
structurally expressive but track noise. The architecture's inductive bias
(complex, relational, rotational, constructive) makes gradient learning more
sample-efficient for the relevant pattern types — it shapes what can be
represented efficiently once the parameters are tuned.

## 14.1 Chunked Parallel Form

The sequential forward can be accelerated during training with a chunked scheme
that amortizes coupling cost across `K`-token chunks. The sequence is divided
into non-overlapping windows of size `K`. Within each chunk, the coupling
operator is evaluated once per token in parallel using a fixed reference state
(the chunk-start state), rather than once per token sequentially against the
evolving state.

For a chunk beginning at position `c`:

**Boundary step** (token `t = c`, exact):

```math
Z_c         = s_c odot S_{c-1}
R_c         = Cpl( Z_c )
S_c         = N( D_c odot S_{c-1} + A_c odot s_c + B_c odot R_c )
```

**Intra-chunk steps** (tokens `t = c+1, ..., c+K-1`, stale-coupling approximation):

```math
Z_t^{stale} = s_t odot S_c                              (parallel over t)
R_t^{stale} = Cpl( Z_t^{stale} )                        (parallel over t)
S_t         = N( D_t odot S_{t-1} + A_t odot s_t + B_t odot R_t^{stale} )
```

The affine recurrence `D odot S_{t-1} + ...` still advances sequentially within
the chunk. The approximation is only in the relational product: `S_c` replaces
`S_{t-1}` in the coupling input `Z_t`. Because `D = sigmoid(D_raw) in (0,1)`,
the state is contractive and the within-chunk drift `||S_{t-1} - S_c||` is
bounded. Self-relation (§7) is boundary-only: it is computed at the exact step
and zeroed within the chunk to preserve the linearity of the intra-chunk
recurrence.

If dynamic gains are enabled, `D_t`, `A_t`, and `B_t` are still recomputed from
each token's own `s_t` inside the chunk. The approximation remains isolated to
the coupling input `Z_t^{stale}` rather than the gain modulation path.

If cross-layer state injection is enabled, layer `\ell` forms its donated
correction from the chunk-end state `S_{tau,end}^{(\ell)}` and broadcasts that
single correction across every position of the recipient chunk in layer
`\ell+1`. This is causal across layers and across chunks, but non-causal within
the chunk: early tokens in the chunk receive a correction that already encodes
later tokens from the same chunk. The exact (`K = None`) path uses the same
whole-span end state, so strict token-level causality still requires
`enable_cross_layer_state=False`.

**Properties:**

- `K = 1` recovers exact sequential stepping; `1 < K < T` enables parallelism.
  The implementation falls back to sequential when `K >= T` because a
  single-chunk stale approximation provides no benefit over exact stepping.
- All five coupling backends (`sequential`, `fft`, `dwt`, `wavelet_packet`,
  `wavelet_packet_max_gauge`) are equally compatible. Each backend operates on
  per-token tensors of shape `(M_1, ..., M_r)`, and the chunked form simply
  evaluates the backend on a batch of `B * (K-1)` such tensors at once.
- The coupling is never absent during training — it is approximated, not
  dropped. There is no train/inference mismatch.
- The gate in the return map (§9) retains full per-token fidelity inside
  chunks: `normalized_hidden` is computed fresh for each token and threaded into
  the gate independently of the coupling approximation.
- Approximation error is controlled by `K`. Smaller `K` gives higher fidelity
  at the cost of less parallelism. The practical `K` is calibrated against the
  coupling contribution to `val_bpc`.

### Chunk drift instrumentation

Training can optionally record a direct diagnostic for the stale-coupling
approximation. For each intra-chunk token, after state normalization, the mixer
measures the batch-mean Frobenius drift from the chunk-start state:

```math
d_t = \mathrm{mean}_{batch} ||S_t - S_c||_F
```

where `c` is the chunk boundary token for the current window. Over a forward
pass the mixer reports:

- `mean_drift`: the mean of `d_t` across all tracked intra-chunk positions
- `max_drift`: the maximum of `d_t`
- `K`: the chunk size used for that pass

This instrumentation is training-only and off by default:

- `chunk_size=None` means exact sequential execution
- `track_chunk_drift=False` means no drift allocation, no per-token scalar
  extraction, and no extra diagnostics in the hot path

At the training-loop level, per-step drift summaries are accumulated and checked
on the same cadence as growth/pruning. If the recent mean chunk drift exceeds
`0.05`, the trainer emits a warning recommending a smaller `K` or exact
sequential execution. This warning is diagnostic only; it does not change model
behavior automatically.

## 14.2 Stateful Sequential Training

The default training loop samples random windows from the corpus — state
resets to the learned initial prior at every batch, so gradient learning never
sees the state that reciprocation learning builds during real streaming.

Stateful training (`stateful_training=True`) aligns the two timescales:

**Corpus streams.** The training corpus is divided into `B` evenly-spaced
streams, one per batch element. Stream `i` starts at position
`i * floor(N / B)` in the token sequence, where `N` is the corpus length.

**Sequential traversal.** Each training step advances every stream forward by
`seq_len` tokens. The batch at step `k` is:

```
inputs[i]  = tokens[ stream_pos[i] : stream_pos[i] + seq_len ]
targets[i] = tokens[ stream_pos[i]+1 : stream_pos[i] + seq_len + 1 ]
stream_pos[i] += seq_len
```

**State carry.** After each optimizer step the per-layer states `S_t^{(l)}`
returned by the forward pass are detached and passed as `initial_state` to
the next step. Detachment truncates gradients at chunk boundaries (standard
truncated BPTT) while preserving the accumulated state value — reciprocation
learning is continuous across chunk boundaries; gradient learning sees a
bounded horizon.

**Wrap reset.** When a stream reaches the end of the corpus it resets to its
start position and its carry state is reset to the learned initial prior. This
breaks continuity at the wrap boundary only.

**Growth events.** When a growth or pruning event rebuilds the model, carry
states are reset to the freshly initialized learned prior for the new state
shape because the state shape has changed.

**Post-rebuild exact cooldown.** If training is using chunked coupling, the
first full growth-check interval after a growth or pruning rebuild is forced
back onto the exact sequential path (`chunk_size=None`). This creates a clean
stabilization window immediately after the state geometry changes, without
changing the configured steady-state chunk size for the rest of training.

The key invariant: GD now trains on the state that reciprocation learning
actually builds, rather than always against a hard zero state that never occurs
at inference time.

## 15. Architectural Parameters

- hidden width `D`
- layer depth `L`
- tensor rank `r` and initial mode sizes `(M_1, ..., M_r)`
- maximum mode sizes and maximum rank (dynamic growth ceilings)
- number of engines per layer: 1 (single mixer)
- normalization choice (Frobenius or per-mode)
- phase scale `pi_phi` (default: `pi`)
- static gain fields `(D_raw, A_raw, B_raw)` with sigmoid/tanh squashing
- optional dynamic gain projector: `dynamic_gains` with `gain_projector_rank`
- optional self-relation gain field `Lambda_self` (zero-initialized)
- coupling type: `sequential`, `fft`, `dwt`, `wavelet_packet`, or
  `wavelet_packet_max_gauge`
- spectral filter parameters: `low_frequency_gain`, `low_frequency_sigma`,
  `high_frequency_gain`, `high_frequency_cutoff`, `wavelet_levels`
- optional dynamic spectral gain projector: `dynamic_spectral_gains` with
  `gain_projector_rank`
- optional FFT anisotropic dynamic gain map: `anisotropic_spectral_gains`
- coupling temperature `tau` (sequential only)
- readout choice (magnitude or phase-aware); phase-aware readout uses a
  learned complex anchor vector of shape `D` (initialized to `1+0i`)
- token magnitude type: `learned`, `inverse_frequency`, or
  `inverse_frequency_learned`
- positional phase type: `rope`, `locked_wave`, or `local_wave`
- token phase: `none`, `semantic`, `virtual_offset`, or
  `semantic_virtual_offset`
- optional anticipator relation (zero-initialized)
- optional cross-layer state injection (zero-initialized gate)
- dynamic growth: `dynamic_mode_growth`, `dynamic_rank_growth`
- dynamic pruning: `dynamic_mode_pruning`, `dynamic_rank_pruning`
- growth thresholds: `growth_residual_threshold`, `residual_saturate_threshold`,
  `rank_growth_loss_ceiling`
- growth EMA decay: `growth_residual_ema_decay`
- growth check interval and minimum checks before first growth
- growth initialization: `mode_init` and `rank_init` (zero, mean, or
  orthogonal/residual)
- pruning guard steps
- training chunk size `K` (§14.1): `None` = exact sequential; `K >= 1` enables
  chunked parallel coupling. Backend-agnostic.
- chunk drift instrumentation (§14.1): `track_chunk_drift=False` by default.
  When enabled, training records per-step `mean_drift` and `max_drift` for the
  current chunk size and warns if recent drift exceeds the fixed `0.05`
  threshold.
- `stateful_training` (§14.2): when enabled, replaces random batch sampling
  with `B` sequential corpus streams; state is carried across chunks via
  truncated BPTT.
- `attention_every_k` (§20): insert a `LocalAttentionBlock` after every k-th
  Reciprocator block. `0` = pure Reciprocator (default).
- `attention_num_heads`: number of attention heads (default 8).
- `attention_window`: KV cache window in tokens (default 256).
- `attention_position`: `"after"` (default) places attention at the end of
  each k-group; `"before"` places it at the start of each k-group (one
  additional attention block for the same `num_layers`).

## 16. Intentional Non-Features

These omissions are design choices, not oversights.

- **No attention by default.** Cross-position transport is carried entirely by
  the tensor-state mixer. Optional causal sliding-window attention blocks can
  be interleaved via `attention_every_k` (§20) for tasks requiring exact
  short-range lookup; the pure Reciprocator is the default.
- **No hard input-gating.** Gains are static parameter tensors with bounded
  parameterizations (sigmoid/tanh). There are no input-dependent gain biases,
  selective saturation, or zero-out mechanisms.
- **No explicit Gram inverse.** The role of `(E^* E + epsilon I)^{-1}` from a
  fixed-dictionary formulation is absorbed into the learned signal-lifting maps
  (`W_mag`, `W_phi`) and downstream weights. There is no analytic overlap
  correction in the forward.
- **No magnitude accumulator.** The return map uses a compact
  phase-aware feature triple `[ Re(c), Im(c), |S| ]`; there is no separate
  accumulator channel or accumulator-modulated gain scaling.
- **No prediction-error drive.** The update has no prediction map `P_k` or
  prediction error term `eta * e_{k,t}`. The state evolves purely through
  decay, input, coupling, and optional self-relation.
- **No engine bank.** Each layer has exactly one mixer with one tensor state.
  There is no inter-engine carry chain.
- **No direct inter-layer state writes.** Even with optional cross-layer state
  injection, layers do not merge, overwrite, or pass ownership of tensor state
  to other layers. The optional path is read-only and hidden-mediated: a layer
  can expose a learned summary of its final state, but each tensor state still
  evolves independently.
- **No fully parallel temporal form.** Training uses sequential token-by-token
  stepping within each chunk. The chunked form (§14.1) parallelizes the coupling
  computation across `K`-token windows but does not eliminate the sequential
  intra-chunk recurrence. A fully parallel prefix-scan form is possible only if
  normalization is deferred to chunk boundaries, which changes model semantics.

---

## 17. Algebraic Structure of the Coupling

### TT analogy: sequential composition, bilinear scoring

The sequential mode coupling (§8) has the left-to-right chain structure of a
tensor-train (TT) contraction. In a standard TT, fixed cores
`G_m[i_m] in C^{R_{m-1} x R_m}` contract as
`X[i_1,...,i_r] = G_1[i_1] * G_2[i_2] * ... * G_r[i_r]`. The sequential mode
coupling shares this dependency pattern: `T_m` is computed from `Z^{(m-1)}`, not
`Z^{(0)}`, so later modes adapt to earlier mixing.

But the coupling is strictly richer than any fixed TT. The bilinear score
`Score_m = (W_m Z_m) Z_m^T` is quadratic in the input-state product
`Z = s_t odot S_{t-1}`. The effective "cores" are not fixed — they are
second-order data-dependent operators. Since `S_{t-1}` is itself a nonlinear
function of `s_{1:t-1}`, the coupling accumulates compositional depth with each
timestep:

```math
Score_m ~ (s_t odot S_{t-1})^2 ~ (s_t * S_{t-1})^2
```

A fixed TT has bounded expressivity determined by its bond dimensions. This
coupling does not: the effective algebraic depth at timestep `t` is `O(t * r)`,
not `r`. The correct framing is a depth-`r` bilinear contraction where the
"cores" are themselves quadratic functions of the incoming signal, and the
signal itself carries the accumulated history of all prior updates.

### State-dependent coupling rank

The coupling's expressivity is bounded by the CP rank of the current state. For
a rank-1 state `S = u_1 (x) ... (x) u_r`, each mode unfolding `Z_m` is rank 1,
so `Score_m = W_m Z_m Z_m^T` is rank 1, and the routing matrix `T_m` is
determined by a single direction in mode-`m` space. The coupling degenerates to
rank-1 routing regardless of `M_m`.

More generally, a state of CP rank `R` supports coupling matrices of effective
rank up to `R`. This is a genuine capacity constraint: the coupling expressivity
grows with the tensor rank of the state. Early in the sequence, the coupling
expressivity is limited by the CP rank already present in the learned initial
prior. Because that prior is trainable and unit-normalized rather than zero,
the model begins with a nontrivial relational seed and a usable global gauge.
This softens the cold-start problem, but does not eliminate it: context-specific
rank must still accumulate online from the input stream, whereas attention is at
full expressivity from position 1.

The state accumulates CP rank over time as inputs perturb it away from rank-1
structure. Normalization modulates this: Frobenius normalization favors unit-norm
states spread across many CP components at low magnitude, while per-mode
normalization constrains each mode independently.

## 18. Norm Geometry and Gradient Flow

### Not unitary: norm-projected dynamics

The coupling matrices are magnitude-row-stochastic times unit-phasor:
`sum_j |(T_m)_{ij}| = 1` per row. This gives `||T_m||_{1->1} <= 1` for the
magnitude matrix, but the spectral norm can reach `sqrt(M_m)`:

```math
||T_m||_2 <= sqrt(||T_m||_inf) <= sqrt(M_m)
```

because column sums can reach `M_m` if all rows concentrate on one column. Over
`r` sequential modes, the coupling can amplify the Frobenius norm by up to
`sqrt(M_state)`. These are not unitary operators.

The boundedness of the dynamics comes from normalization (§5), which projects
the state onto the unit sphere in `C^{M_state}` (Frobenius case) or a product of
fiber-wise unit spheres (per-mode case) after each update. The coupling step
itself is not norm-preserving; normalization enforces boundedness by projection.

The correct geometric characterization: the dynamics live on complex projective
space `CP^{M_state - 1}` (Frobenius) or a product of complex unit spheres
(per-mode). The coupling is a potentially expansive map on projective space;
normalization clips the expansion.

### Gradient implications

Between normalization steps, the Jacobian of the coupling can have singular
values up to `sqrt(M_m)` per mode. The normalization Jacobian has singular
values scaling as `1/||S_tilde||_F`, which compensates but does not cancel the
coupling amplification in a clean algebraic way. This is not the clean gradient
highway that true unitary evolution provides (cf. Arjovsky et al. 2016, Wisdom
et al. 2016).

The practical consequence: gradient flow through many sequential coupling steps
may encounter ill-conditioned Jacobians, particularly when the softmax
concentrates (small `tau`). Large `tau` makes the softmax near-uniform and the
coupling nearer to non-expansive, at the cost of reduced selectivity. The
temperature `tau` thus trades routing expressivity against gradient conditioning.

## 19. Expressivity vs. Attention

The fundamental tradeoff between the Reciprocator's tensor-state mixer and
attention:

| Property | Attention (`T` positions, `d` dim) | Reciprocator (state `(M_1,...,M_r)`) |
|---|---|---|
| Compute per step | `O(T * d)` | `O(M_state * max M_m^2)` |
| Memory per step | `O(T * d)` | `O(M_state)` |
| Streaming | No (requires full prefix) | Yes (true `O(1)` incremental) |
| Effective history rank | `<= d` per head (softmax rank) | `<=` CP rank of state |
| Positional access | Exact (attend to position `i`) | Compressed (state is lossy summary) |
| Algebraic depth at step `t` | 2 (QK bilinear + V mixing) | `O(t * r)` (accumulated composition) |
| Cold-start | Full expressivity from position 1 | Starts from a learned prior; still needs early tokens to build context-specific CP rank |

Attention trades `O(T)` compute for exact positional access; the Reciprocator
trades `O(1)` compute for compressed access. The compression quality depends on
whether the relevant history has low tensor rank.

**When the tensor structure wins:** compositional or factorizable structure in
the data (e.g., syntax trees where joint dependencies factor across discrete
structural modes), streaming or infinite-context requirements, or when the
effective attention rank is low enough that a tensor summary captures most of
the information.

**When it loses:** tasks requiring precise positional lookup within long
sequences (the compressed state cannot recover exact positions), high-rank
attention patterns (where no low-CP-rank summary is adequate), or when the
optimal tensor factorization for the task is unknown and the model must discover
it.

## 20. Hybrid Architecture: Interleaved Attention

The Reciprocator's compressed state trades O(1) incremental compute for
lossy positional access: the state is a sufficient statistic for the history,
not an exact record. Tasks requiring precise short-range lookup — variable
bindings in code, equation references in math — expose this limitation.

The hybrid architecture adds causal sliding-window attention blocks
interleaved with Reciprocator blocks on the shared hidden stream. The two
mechanisms are complementary:

| Mechanism | Role | Horizon | Cost per step |
|---|---|---|---|
| Reciprocator mixer | Long-range pattern accumulation | Unbounded (state) | O(M_state) |
| Local attention | Exact short-range lookup | Window W | O(W · D) |

### Block layout

Attention blocks are inserted after every k-th Reciprocator block, except
after the final block. With `num_layers=12, attention_every_k=3`:

```
Recip Recip Recip Attn Recip Recip Recip Attn Recip Recip Recip Attn Recip Recip Recip
```

`attention_every_k=0` (default) disables attention entirely, recovering the
pure Reciprocator.

### Attention block

Each `LocalAttentionBlock` is fully complex. Q, K, V are projected from the
complex hidden stream via `ComplexLinear`:

```math
Q_t = W_Q * cLN(h_t),   K_t = W_K * cLN(h_t),   V_t = W_V * cLN(h_t)    in C^{n_heads x d_head}
```

Attention scores use the **Hermitian inner product** — real-valued and
phase-sensitive:

```math
score(i, j) = Re( Q_i K_j^dagger ) / sqrt(d_head)
```

`Re(Q K†)` measures how phase-aligned and magnitude-similar the complex query
and key vectors are. Queries and keys that differ only by a global phase shift
score identically; those with opposing phases score negatively. This is the
natural inner product for the complex projective geometry the Reciprocator
operates in.

If a KV cache from the previous chunk is present, its tokens are prepended as
a fully-visible prefix; the causal mask applies only within the current chunk:

```math
K = [ K_cache ; K_t ],   V = [ V_cache ; V_t ]
mask[i, j] = begin{cases} 0 & j < cache_len \\ -inf & j >= cache_len + i \end{cases}
```

Softmax weights are real. They are applied to the **complex** V, so phase
flows through unmodified:

```math
out_i = sum_j softmax(score)_{ij} * V_j     in C^{n_heads x d_head}
delta_t = W_out * out_t                       in C^D
h_t = h_t + delta_t
```

The KV cache is windowed to `attention_window` tokens (default 256). Tokens
beyond the window are dropped from the oldest end.

### State threading

`LocalAttentionBlock` carries a complex KV cache `(K, V)` — dtype `cfloat`,
shape `[batch, cache_len, heads, head_dim]` — as its block state, using the
same interface as `ReciprocatorBlock`. During stateful training, the KV cache
is detached and carried across chunk boundaries alongside the Reciprocator
tensor states. On stream wrap or growth events, the KV cache is zeroed like
all other carry states.

### Recommended configuration for coding and math

```python
TrainingConfig(
    hidden_size=512,
    num_layers=12,
    ffn_expansion_factor=4,
    state_shape=(8, 8, 4),
    normalization_type="per_mode",
    enable_self_relation=True,
    attention_every_k=3,
    attention_num_heads=8,
    attention_window=256,
    token_magnitude_type="inverse_frequency_learned",
    phase_type="rope",
    readout_type="phase_aware",
    stateful_training=True,
    seq_len=256,
)
```

To enable relational gain modulation, add `dynamic_gains=True`. The low-rank
projector width is controlled by `gain_projector_rank` (default `8`).

To enable adaptive spectral filters, use a spectral `coupling_type` and set
`dynamic_spectral_gains=True`; the CLI flag is `--dynamic-spectral-gains`. This
uses the same `gain_projector_rank` width as dynamic mixer gains.
For FFT, add `anisotropic_spectral_gains=True` or
`--anisotropic-spectral-gains` to replace radial dynamic modulation with a full
coordinatewise frequency-grid map. Do not treat `anisotropic_spectral_gains=True`
alone as a separate operator; without `dynamic_spectral_gains=True` it is an
inert control that should match fixed radial FFT.

## 21. Growth as Adaptive-Rank Sketching

The growth rule (§12) is not streaming CP decomposition. Streaming CP observes
tensor `X_t`, maintains decomposition `X_hat_t = sum lambda_i a_i(t) (x) b_i (x) c_i`,
and increases rank `R` when `||X_t - X_hat_t|| > epsilon`. The residual is in
data space — you know exactly what you're failing to represent.

The Reciprocator growth rule computes the residual of the signal's mode-`m`
marginals against the gain-parameter row space `{D_logit, A_logit, B_logit}`,
and grows mode `m` when this residual exceeds a threshold. The residual is in
**parameter space** — you know what your parameters can't express, but not
whether that expressiveness is needed for the loss.

This is closer to adaptive-rank matrix sketching (e.g., Liberty's frequent
directions):

1. Maintain a fixed-size sketch (the gain parameters) of the signal stream.
2. Monitor whether the sketch captures the signal's directional variation.
3. When the residual grows, increase the sketch size (add a mode slice or rank).

The `_orthogonalize_candidate` initialization is exactly the residual-direction
initialization from adaptive-rank methods: project the candidate onto the
orthogonal complement of the existing row space and use that as the new
direction. This preserves what's already learned while adding capacity for the
new direction.

**Key difference that matters:** in streaming CP, increasing the rank improves
the approximation of the observed data. In the Reciprocator, increasing the
mode size improves the capacity of the gain parameters to modulate future
updates. These are not the same thing. The growth rule optimizes for parameter
expressiveness, not reconstruction quality. It is possible to grow in directions
that don't help the downstream loss.

The pruning rule (§13) is the dual and is cleaner. It measures how redundant
each mode is with respect to the other modes — asking whether mode `m` carries
information not already captured by the other modes. If not, it is pruned by
averaging. This is closer to classical tensor rank reduction (merging redundant
modes), and the criterion is well-motivated.
