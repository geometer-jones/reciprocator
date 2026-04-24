import pytest
import torch

from reciprocator.complex_ops import normalize_complex
from reciprocator.mixer import ReciprocatorMixer, WaveletPacketSpectralCoupling, _WaveletNode


def random_complex(*shape: int) -> torch.Tensor:
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


def assert_finite(tensor: torch.Tensor) -> None:
    if torch.is_complex(tensor):
        assert torch.isfinite(tensor.real).all()
        assert torch.isfinite(tensor.imag).all()
    else:
        assert torch.isfinite(tensor).all()


def force_return_map_to_first_magnitude(mixer: ReciprocatorMixer) -> None:
    with torch.no_grad():
        mixer.return_map.proj.weight.zero_()
        mixer.return_map.proj.bias.zero_()
        mixer.return_map.proj.weight[0, 2 * mixer.state_size] = 1.0
        mixer.gate.weight.zero_()
        mixer.gate.bias.fill_(20.0)


def gated_delta_from_features(
    mixer: ReciprocatorMixer,
    features: torch.Tensor,
    normalized_hidden: torch.Tensor,
) -> torch.Tensor:
    raw_delta = mixer.return_map(features)
    gate_input = torch.cat(
        [normalized_hidden.real, normalized_hidden.imag, raw_delta.real, raw_delta.imag], dim=-1
    )
    gate = torch.sigmoid(mixer.gate(gate_input))
    return gate.to(raw_delta.dtype) * raw_delta


@pytest.mark.parametrize(
    "coupling_type",
    ["sequential", "fft", "dwt", "wavelet_packet", "wavelet_packet_max_gauge"],
)
def test_mixer_preserves_shapes_and_normalizes_state(coupling_type: str) -> None:
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3), coupling_type=coupling_type)
    hidden = random_complex(2, 5, 8)

    delta, state = mixer(hidden)

    assert delta.shape == (2, 5, 8)
    assert state.shape == (2, 2, 3)

    norms = torch.sqrt((state.real.square() + state.imag.square()).sum(dim=(1, 2)))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_mixer_supports_per_mode_state_normalization() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3), normalization_type="per_mode")
    hidden = random_complex(2, 5, 8)

    _, state = mixer(hidden)

    power = state.real.square() + state.imag.square()
    mode_1_norms = torch.sqrt(power.sum(dim=1))
    mode_2_norms = torch.sqrt(power.sum(dim=2))
    expected_mode_1 = torch.full_like(mode_1_norms, (2.0 / 3.0) ** 0.5)
    expected_mode_2 = torch.ones_like(mode_2_norms)

    assert torch.allclose(mode_1_norms, expected_mode_1, atol=2e-3)
    assert torch.allclose(mode_2_norms, expected_mode_2, atol=1e-5)


@pytest.mark.parametrize(
    "coupling_type",
    ["sequential", "fft", "dwt", "wavelet_packet", "wavelet_packet_max_gauge"],
)
def test_streaming_matches_full_sequence(coupling_type: str) -> None:
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2), coupling_type=coupling_type)
    hidden = random_complex(3, 4, 6)

    batch_delta, batch_state = mixer(hidden)

    step_state = mixer.initial_state(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
    step_outputs = []
    for index in range(hidden.shape[1]):
        delta_t, step_state = mixer.step(hidden[:, index], step_state)
        step_outputs.append(delta_t)
    streamed_delta = torch.stack(step_outputs, dim=1)

    assert torch.allclose(streamed_delta, batch_delta, atol=1e-5)
    assert torch.allclose(step_state, batch_state, atol=1e-5)


def test_future_tokens_do_not_change_prefix_outputs() -> None:
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2))
    prefix = random_complex(2, 3, 6)
    suffix_a = random_complex(2, 2, 6)
    suffix_b = random_complex(2, 2, 6)

    seq_a = torch.cat([prefix, suffix_a], dim=1)
    seq_b = torch.cat([prefix, suffix_b], dim=1)

    delta_a, _ = mixer(seq_a)
    delta_b, _ = mixer(seq_b)

    assert torch.allclose(delta_a[:, : prefix.shape[1]], delta_b[:, : prefix.shape[1]], atol=1e-5)


def test_spectral_couplings_produce_distinct_updates() -> None:
    torch.manual_seed(0)
    relational = random_complex(2, 2, 4)
    outputs = {}

    for coupling_type in ("fft", "dwt", "wavelet_packet"):
        mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 4), coupling_type=coupling_type)
        outputs[coupling_type] = mixer.coupling(relational)

    assert not torch.allclose(outputs["fft"], outputs["dwt"], atol=1e-5)
    assert not torch.allclose(outputs["fft"], outputs["wavelet_packet"], atol=1e-5)
    assert not torch.allclose(outputs["dwt"], outputs["wavelet_packet"], atol=1e-5)


def test_wavelet_packet_max_gauge_enables_phase_aware_basis_selection() -> None:
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 3), coupling_type="wavelet_packet_max_gauge")

    assert mixer.coupling.phase_aware_best_basis is True


@pytest.mark.parametrize("coupling_type", ["fft", "dwt", "wavelet_packet", "wavelet_packet_max_gauge"])
def test_dynamic_spectral_gains_are_inert_at_initialization(coupling_type: str) -> None:
    tensor = random_complex(2, 2, 4)
    static_mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 4), coupling_type=coupling_type)
    dynamic_mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(2, 4),
        coupling_type=coupling_type,
        dynamic_spectral_gains=True,
        gain_projector_rank=4,
    )

    static_output = static_mixer.coupling(tensor)
    dynamic_output = dynamic_mixer.coupling(tensor)

    assert torch.allclose(dynamic_output, static_output, atol=1e-6)
    assert dynamic_mixer.coupling.dynamic_spectral_gains is True
    assert dynamic_mixer.coupling.spectral_projector is not None


@pytest.mark.parametrize("coupling_type", ["fft", "dwt", "wavelet_packet", "wavelet_packet_max_gauge"])
def test_dynamic_spectral_gains_can_change_profile(coupling_type: str) -> None:
    tensor = random_complex(2, 2, 4)
    mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(2, 4),
        coupling_type=coupling_type,
        dynamic_spectral_gains=True,
        gain_projector_rank=4,
    )
    baseline = mixer.coupling(tensor)

    with torch.no_grad():
        mixer.coupling.alpha_spectral.fill_(0.25)
        mixer.coupling.spectral_projector[-1].bias.copy_(torch.linspace(0.1, 0.5, steps=8))

    adapted = mixer.coupling(tensor)

    assert not torch.allclose(adapted, baseline, atol=1e-6)
    assert_finite(adapted)


def test_dynamic_spectral_gains_receive_projector_gradients_from_inert_init() -> None:
    tensor = random_complex(2, 2, 4).requires_grad_(True)
    mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(2, 4),
        coupling_type="fft",
        dynamic_spectral_gains=True,
        gain_projector_rank=4,
    )

    output = mixer.coupling(tensor)
    loss = output.abs().square().sum()
    loss.backward()

    projector_head = mixer.coupling.spectral_projector[-1]
    assert projector_head.weight.grad is not None
    assert projector_head.bias.grad is not None
    assert projector_head.weight.grad.abs().sum().item() > 0.0
    assert projector_head.bias.grad.abs().sum().item() > 0.0


def test_anisotropic_fft_spectral_gains_distinguish_equal_radius_coordinates() -> None:
    tensor = random_complex(2, 3, 3)
    radial_mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(3, 3),
        coupling_type="fft",
        dynamic_spectral_gains=True,
        gain_projector_rank=4,
    )
    anisotropic_mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(3, 3),
        coupling_type="fft",
        dynamic_spectral_gains=True,
        anisotropic_spectral_gains=True,
        gain_projector_rank=4,
    )
    frequency_bands = [torch.fft.fftfreq(size, device=tensor.device) for size in tensor.shape[1:]]
    mesh = torch.meshgrid(*frequency_bands, indexing="ij")
    radius = torch.sqrt(sum(component.square() for component in mesh))
    radial_frequency = radius / radius.max().clamp_min(1e-8)

    with torch.no_grad():
        for mixer in (radial_mixer, anisotropic_mixer):
            mixer.coupling.alpha_spectral.fill_(1.0)
            mixer.coupling.spectral_projector[-1].bias.copy_(torch.linspace(0.1, 0.9, steps=9))

    radial_gain = radial_mixer.coupling._spectral_gain(radial_frequency, signal=tensor)
    anisotropic_gain = anisotropic_mixer.coupling._spectral_gain(radial_frequency, signal=tensor)

    assert radial_mixer.coupling.anisotropic_spectral_gains is False
    assert anisotropic_mixer.coupling.anisotropic_spectral_gains is True
    assert torch.allclose(radial_gain[:, 0, 1], radial_gain[:, 1, 0], atol=1e-6)
    assert not torch.allclose(anisotropic_gain[:, 0, 1], anisotropic_gain[:, 1, 0], atol=1e-6)


def test_wavelet_packet_phase_coherence_handles_zero_and_tiny_coefficients() -> None:
    coupling = WaveletPacketSpectralCoupling(
        state_shape=(2, 2),
        low_frequency_gain=0.5,
        low_frequency_sigma=0.35,
        high_frequency_gain=0.5,
        high_frequency_cutoff=0.5,
        wavelet_levels=None,
        phase_aware_best_basis=True,
    )
    zero_coeffs = torch.zeros(2, 4, dtype=torch.cfloat)
    tiny_coeffs = torch.complex(
        torch.tensor([[0.0, 1e-30, -1e-30, 0.0], [1e-30, 0.0, 0.0, -1e-30]]),
        torch.tensor([[0.0, -1e-30, 1e-30, 0.0], [0.0, 1e-30, -1e-30, 0.0]]),
    )

    zero_coherence = coupling._phase_coherence(zero_coeffs, torch.ones(2, 1, dtype=torch.cfloat))
    tiny_coherence = coupling._phase_coherence(tiny_coeffs, torch.zeros(2, 1, dtype=torch.cfloat))
    reconstructed = coupling(tiny_coeffs.reshape(2, 2, 2))

    assert_finite(zero_coherence)
    assert_finite(tiny_coherence)
    assert_finite(reconstructed)
    assert zero_coherence.item() == pytest.approx(1.0)
    assert 0.0 <= tiny_coherence.item() <= 1.0


@pytest.mark.parametrize("operation", ["best_basis", "filter", "reconstruct"])
def test_wavelet_packet_malformed_internal_node_raises_value_error(operation: str) -> None:
    coupling = WaveletPacketSpectralCoupling(
        state_shape=(2, 2),
        low_frequency_gain=0.5,
        low_frequency_sigma=0.35,
        high_frequency_gain=0.5,
        high_frequency_cutoff=0.5,
        wavelet_levels=None,
        phase_aware_best_basis=False,
    )
    coeffs = random_complex(1, 4)
    malformed_node = _WaveletNode(
        coeffs=coeffs,
        original_length=4,
        band_low=0.0,
        band_high=1.0,
        approx=_WaveletNode(
            coeffs=coeffs[..., :2],
            original_length=2,
            band_low=0.0,
            band_high=0.5,
        ),
        detail=None,
    )

    with pytest.raises(ValueError, match="Wavelet packet tree invariant"):
        if operation == "best_basis":
            coupling._select_best_basis(malformed_node, torch.ones(1, 1, dtype=torch.cfloat))
        elif operation == "filter":
            coupling._filter_leaves(malformed_node)
        else:
            coupling._reconstruct(malformed_node)


def test_dynamic_gains_flag_inconsistency_raises_value_error() -> None:
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2))
    mixer.enable_dynamic_gains = True

    with pytest.raises(ValueError, match="gain_projector"):
        mixer.step(random_complex(2, 6))


@pytest.mark.parametrize(
    "coupling_type",
    ["sequential", "fft", "dwt", "wavelet_packet", "wavelet_packet_max_gauge"],
)
def test_saturated_static_gains_remain_finite(coupling_type: str) -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(2, 2),
        enable_self_relation=True,
        coupling_type=coupling_type,
    )
    with torch.no_grad():
        mixer.decay_logit.fill_(80.0)
        mixer.input_logit.fill_(-80.0)
        mixer.recurrent_logit.fill_(80.0)
        mixer.self_relation_logit.fill_(80.0)

    hidden_t = random_complex(3, 6) * 1e12
    state = random_complex(3, 2, 2) * 1e12

    delta, next_state = mixer.step(hidden_t, state)

    assert_finite(delta)
    assert_finite(next_state)
    norms = torch.sqrt((next_state.real.square() + next_state.imag.square()).sum(dim=(1, 2)))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_saturated_dynamic_gains_remain_finite() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(2, 2),
        enable_dynamic_gains=True,
        gain_projector_rank=4,
    )
    with torch.no_grad():
        mixer.alpha_D.fill_(100.0)
        mixer.alpha_A.fill_(100.0)
        mixer.alpha_B.fill_(100.0)
        mixer.gain_projector[-1].bias.fill_(100.0)

    hidden_t = random_complex(2, 6) * 1e12
    state = random_complex(2, 2, 2) * 1e12

    delta, next_state = mixer.step(hidden_t, state)

    assert_finite(delta)
    assert_finite(next_state)
    norms = torch.sqrt((next_state.real.square() + next_state.imag.square()).sum(dim=(1, 2)))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.parametrize("scale", [1e-18, 1e18])
@pytest.mark.parametrize(
    "coupling_type",
    ["sequential", "fft", "dwt", "wavelet_packet", "wavelet_packet_max_gauge"],
)
def test_extreme_hidden_and_state_amplitudes_remain_finite(coupling_type: str, scale: float) -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2), coupling_type=coupling_type)
    hidden = random_complex(2, 3, 6) * scale
    state = random_complex(2, 2, 2) * scale

    delta, next_state = mixer(hidden, state)

    assert_finite(delta)
    assert_finite(next_state)
    norms = torch.sqrt((next_state.real.square() + next_state.imag.square()).sum(dim=(1, 2)))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_self_relation_flag_is_inert_at_initialization() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2))
    self_related_mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2), enable_self_relation=True)
    incompatible_keys = self_related_mixer.load_state_dict(mixer.state_dict(), strict=False)

    assert incompatible_keys.missing_keys == ["self_relation_logit"]
    assert incompatible_keys.unexpected_keys == []

    hidden_t = random_complex(2, 6)
    state = random_complex(2, 2, 2)

    delta, next_state = mixer.step(hidden_t, state)
    self_related_delta, self_related_state = self_related_mixer.step(hidden_t, state)

    assert torch.allclose(self_related_delta, delta, atol=1e-5)
    assert torch.allclose(self_related_state, next_state, atol=1e-5)


def test_self_relation_flag_changes_update_when_gain_is_nonzero() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2))
    self_related_mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2), enable_self_relation=True)
    self_related_mixer.load_state_dict(mixer.state_dict(), strict=False)

    with torch.no_grad():
        self_related_mixer.self_relation_logit.fill_(2.0)

    hidden_t = random_complex(2, 6)
    state = random_complex(2, 2, 2)

    delta, next_state = mixer.step(hidden_t, state)
    self_related_delta, self_related_state = self_related_mixer.step(hidden_t, state)

    assert not torch.allclose(self_related_delta, delta, atol=1e-5)
    assert not torch.allclose(self_related_state, next_state, atol=1e-5)


def test_cross_bilinear_disabled_matches_baseline() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(2, 2),
        enable_cross_bilinear=False,
    )
    torch.manual_seed(0)
    cross_mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(2, 2),
        enable_cross_bilinear=True,
        cross_bilinear_rank=3,
    )
    hidden = random_complex(2, 4, 6)

    with torch.no_grad():
        mixer.recurrent_logit.fill_(1.0)
        cross_mixer.recurrent_logit.fill_(1.0)

    delta, state = mixer(hidden)
    cross_delta, cross_state = cross_mixer(hidden)

    assert mixer.cross_bilinear is None
    assert cross_mixer.cross_bilinear is not None
    assert torch.allclose(cross_delta, delta, atol=0.0, rtol=0.0)
    assert torch.allclose(cross_state, state, atol=0.0, rtol=0.0)


def test_cross_bilinear_is_enabled_by_default() -> None:
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2))

    assert mixer.enable_cross_bilinear is True
    assert mixer.cross_bilinear is not None


def test_cross_bilinear_gradients_flow() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(
        hidden_size=6,
        state_shape=(2, 2),
        enable_cross_bilinear=True,
        cross_bilinear_rank=3,
    )
    assert mixer.cross_bilinear is not None
    bilinear = mixer.cross_bilinear
    optimizer = torch.optim.SGD(bilinear.parameters(), lr=0.1)
    z_flat = random_complex(5, mixer.state_size)
    target = random_complex(5, mixer.state_size)

    loss = (bilinear(z_flat) - target).abs().square().sum()
    loss.backward()

    assert bilinear.W.weight_real.grad is not None
    assert bilinear.W.weight_real.grad.abs().sum().item() > 0.0

    optimizer.step()
    optimizer.zero_grad()

    second_target = random_complex(5, mixer.state_size)
    second_loss = (bilinear(z_flat) - second_target).abs().square().sum()
    second_loss.backward()

    for projection in (bilinear.U, bilinear.V, bilinear.W):
        grad_magnitude = projection.weight_real.grad.abs().sum()
        grad_magnitude = grad_magnitude + projection.weight_imag.grad.abs().sum()
        assert grad_magnitude.item() > 0.0


def test_cross_bilinear_chunked_matches_sequential() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(
        hidden_size=8,
        state_shape=(2, 3),
        enable_cross_bilinear=True,
        cross_bilinear_rank=4,
    )
    assert mixer.cross_bilinear is not None
    hidden = random_complex(2, 8, 8)

    with torch.no_grad():
        mixer.recurrent_logit.fill_(1.0)
        mixer.cross_bilinear.W.weight_real.normal_(mean=0.0, std=0.02)
        mixer.cross_bilinear.W.weight_imag.normal_(mean=0.0, std=0.02)

    out_seq, state_seq = mixer(hidden)
    out_k1, state_k1 = mixer(hidden, chunk_size=1)

    assert torch.allclose(out_k1, out_seq, atol=0.0)
    assert torch.allclose(state_k1, state_seq, atol=0.0)


@pytest.mark.parametrize("state_shape", [(4,), (3, 4), (2, 3, 4)])
def test_cross_bilinear_shapes(state_shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(
        hidden_size=8,
        state_shape=state_shape,
        enable_cross_bilinear=True,
        cross_bilinear_rank=4,
    )
    assert mixer.cross_bilinear is not None
    hidden = random_complex(2, 5, 8)

    with torch.no_grad():
        mixer.recurrent_logit.fill_(1.0)
        mixer.cross_bilinear.W.weight_real.normal_(mean=0.0, std=0.02)
        mixer.cross_bilinear.W.weight_imag.normal_(mean=0.0, std=0.02)

    out, state = mixer(hidden)
    chunked_out, chunked_state = mixer(hidden, chunk_size=2)

    assert out.shape == hidden.shape
    assert state.shape == (hidden.shape[0], *state_shape)
    assert chunked_out.shape == hidden.shape
    assert chunked_state.shape == (hidden.shape[0], *state_shape)


def test_return_map_features_are_phase_invariant_and_compact() -> None:
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 3))
    state = random_complex(2, 2, 3)
    global_phase = torch.polar(torch.ones(1, 1, 1), torch.tensor(0.73))

    features = mixer._state_features(state)
    rotated_features = mixer._state_features(state * global_phase)

    assert mixer.return_map.proj.in_features == 3 * mixer.state_size
    assert features.shape == (2, 3 * mixer.state_size)
    assert torch.allclose(rotated_features, features, atol=1e-5)


def test_step_return_map_uses_state_displacement_features() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2))
    force_return_map_to_first_magnitude(mixer)
    hidden_t = random_complex(2, 6)
    state = random_complex(2, 2, 2)

    delta, next_state = mixer.step(hidden_t, state)

    normalized_hidden = mixer.pre_norm(hidden_t)
    expected = gated_delta_from_features(
        mixer,
        mixer._state_features(next_state - state),
        normalized_hidden,
    )
    absolute_state_delta = gated_delta_from_features(
        mixer,
        mixer._state_features(next_state),
        normalized_hidden,
    )

    assert torch.allclose(delta, expected, atol=1e-5)
    assert not torch.allclose(delta, absolute_state_delta, atol=1e-5)


def test_compute_delta_uses_state_displacement_features() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2))
    force_return_map_to_first_magnitude(mixer)
    hidden_t = random_complex(2, 6)
    old_state = random_complex(2, 2, 2)
    next_state = random_complex(2, 2, 2)
    normalized_hidden = mixer.pre_norm(hidden_t)

    delta = mixer._compute_delta(next_state, old_state, normalized_hidden)

    expected = gated_delta_from_features(
        mixer,
        mixer._state_features(next_state - old_state),
        normalized_hidden,
    )
    absolute_state_delta = gated_delta_from_features(
        mixer,
        mixer._state_features(next_state),
        normalized_hidden,
    )

    assert torch.allclose(delta, expected, atol=1e-5)
    assert not torch.allclose(delta, absolute_state_delta, atol=1e-5)


# --- chunked forward ---

CHUNKED_COUPLING_TYPES = ["sequential", "fft", "dwt", "wavelet_packet", "wavelet_packet_max_gauge"]


def _manual_chunk_drifts(mixer: ReciprocatorMixer, hidden: torch.Tensor, chunk_size: int) -> list[float]:
    state = mixer.initial_state(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
    drifts = []

    for start in range(0, hidden.shape[1], chunk_size):
        end = min(start + chunk_size, hidden.shape[1])
        chunk = hidden[:, start:end]

        _, state = mixer.step(chunk[:, 0], state)
        chunk_start_state = state

        for index in range(1, chunk.shape[1]):
            normed = mixer.pre_norm(chunk[:, index])
            signal = mixer.signal_projector(normed)
            routed_stale = mixer.coupling(signal * chunk_start_state)
            decay_logit, input_logit, recurrent_logit = mixer._gain_logits(signal)
            proposal = (
                torch.sigmoid(decay_logit) * state
                + torch.sigmoid(input_logit) * signal
                + torch.tanh(recurrent_logit) * routed_stale
            )
            state = normalize_complex(
                proposal,
                normalization_type=mixer.normalization_type,
                dims=tuple(range(1, proposal.ndim)),
                eps=mixer.eps,
            )
            drift = (state - chunk_start_state).flatten(start_dim=1).norm(p=2, dim=1).mean()
            drifts.append(float(drift.item()))

    return drifts


@pytest.mark.parametrize("coupling_type", CHUNKED_COUPLING_TYPES)
def test_chunked_k1_matches_sequential(coupling_type: str) -> None:
    # chunk_size=1 means every token is a boundary step; must be bit-exact with sequential
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3), coupling_type=coupling_type)
    hidden = random_complex(2, 8, 8)

    out_seq, state_seq = mixer(hidden)
    out_k1, state_k1 = mixer(hidden, chunk_size=1)

    assert torch.allclose(out_k1, out_seq, atol=0.0)
    assert torch.allclose(state_k1, state_seq, atol=0.0)


@pytest.mark.parametrize("coupling_type", CHUNKED_COUPLING_TYPES)
def test_chunked_output_shapes(coupling_type: str) -> None:
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3), coupling_type=coupling_type)
    hidden = random_complex(2, 12, 8)

    for chunk_size in (3, 4, 6, 12):
        out, state = mixer(hidden, chunk_size=chunk_size)
        assert out.shape == (2, 12, 8), f"chunk_size={chunk_size}: wrong output shape"
        assert state.shape == (2, 2, 3), f"chunk_size={chunk_size}: wrong state shape"


def test_chunked_non_divisible_sequence_length() -> None:
    # T=10 is not divisible by K=4; all T tokens must still be produced
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3))
    hidden = random_complex(2, 10, 8)

    out, state = mixer(hidden, chunk_size=4)

    assert out.shape == (2, 10, 8)
    assert state.shape == (2, 2, 3)


def test_chunked_large_chunk_falls_back_to_sequential() -> None:
    # chunk_size >= T triggers the sequential fallback path
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3))
    hidden = random_complex(2, 6, 8)

    out_seq, state_seq = mixer(hidden)
    out_large, state_large = mixer(hidden, chunk_size=100)

    assert torch.allclose(out_large, out_seq, atol=0.0)
    assert torch.allclose(state_large, state_seq, atol=0.0)


def test_chunked_none_chunk_size_matches_sequential() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3))
    hidden = random_complex(2, 6, 8)

    out_seq, state_seq = mixer(hidden)
    out_none, state_none = mixer(hidden, chunk_size=None)

    assert torch.allclose(out_none, out_seq, atol=0.0)
    assert torch.allclose(state_none, state_seq, atol=0.0)


def test_chunked_final_state_is_normalized() -> None:
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3))
    hidden = random_complex(2, 9, 8)

    _, state = mixer(hidden, chunk_size=4)

    norms = torch.sqrt((state.real.square() + state.imag.square()).sum(dim=(1, 2)))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_chunked_track_drift_returns_stats() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3))
    hidden = random_complex(2, 8, 8)

    out, state, drift_stats = mixer(hidden, chunk_size=4, track_drift=True)

    assert out.shape == (2, 8, 8)
    assert state.shape == (2, 2, 3)
    assert drift_stats is not None
    assert drift_stats["K"] == 4
    assert drift_stats["mean_drift"] >= 0.0
    assert drift_stats["max_drift"] >= drift_stats["mean_drift"]


def test_chunked_track_drift_supports_higher_rank_states() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3, 4))
    hidden = random_complex(2, 8, 8)

    out, state, drift_stats = mixer(hidden, chunk_size=4, track_drift=True)

    assert out.shape == (2, 8, 8)
    assert state.shape == (2, 2, 3, 4)
    assert drift_stats is not None
    assert drift_stats["K"] == 4
    assert drift_stats["mean_drift"] >= 0.0
    assert drift_stats["max_drift"] >= drift_stats["mean_drift"]


def test_chunked_track_drift_matches_manual_bounded_drift() -> None:
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=8, state_shape=(2, 3))
    hidden = random_complex(2, 8, 8)

    _, _, drift_stats = mixer(hidden, chunk_size=4, track_drift=True)
    manual_drifts = _manual_chunk_drifts(mixer, hidden, chunk_size=4)

    assert drift_stats is not None
    assert manual_drifts
    assert max(manual_drifts) > 0.0
    # Default Frobenius normalization keeps each state on the unit sphere, so
    # the distance from any chunk-start state is bounded by the diameter.
    assert max(manual_drifts) <= 2.0 + 1e-6
    assert drift_stats["mean_drift"] == pytest.approx(sum(manual_drifts) / len(manual_drifts))
    assert drift_stats["max_drift"] == pytest.approx(max(manual_drifts))


def test_chunked_preserves_causality() -> None:
    # Prefix outputs must be identical regardless of what follows in the sequence
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2))
    prefix = random_complex(2, 4, 6)
    suffix_a = random_complex(2, 4, 6)
    suffix_b = random_complex(2, 4, 6)

    out_a, _ = mixer(torch.cat([prefix, suffix_a], dim=1), chunk_size=4)
    out_b, _ = mixer(torch.cat([prefix, suffix_b], dim=1), chunk_size=4)

    assert torch.allclose(out_a[:, : prefix.shape[1]], out_b[:, : prefix.shape[1]], atol=1e-5)


def test_chunked_self_relation_is_boundary_only() -> None:
    # With nonzero self_relation_logit, chunked and sequential differ because
    # self-relation is zeroed inside chunks. Shapes must still be correct.
    torch.manual_seed(0)
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 2), enable_self_relation=True)
    with torch.no_grad():
        mixer.self_relation_logit.fill_(2.0)

    hidden = random_complex(2, 8, 6)

    out_seq, _ = mixer(hidden)
    out_chunk, _ = mixer(hidden, chunk_size=4)

    assert out_chunk.shape == out_seq.shape
    # Intra-chunk tokens skip self-relation so outputs differ from sequential
    assert not torch.allclose(out_chunk, out_seq, atol=1e-5)
