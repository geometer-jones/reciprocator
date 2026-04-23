import pytest
import torch

from reciprocator.mixer import ReciprocatorMixer


def random_complex(*shape: int) -> torch.Tensor:
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


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


def test_return_map_features_are_phase_invariant_and_compact() -> None:
    mixer = ReciprocatorMixer(hidden_size=6, state_shape=(2, 3))
    state = random_complex(2, 2, 3)
    global_phase = torch.polar(torch.ones(1, 1, 1), torch.tensor(0.73))

    features = mixer._state_features(state)
    rotated_features = mixer._state_features(state * global_phase)

    assert mixer.return_map.proj.in_features == 3 * mixer.state_size
    assert features.shape == (2, 3 * mixer.state_size)
    assert torch.allclose(rotated_features, features, atol=1e-5)


# --- chunked forward ---

CHUNKED_COUPLING_TYPES = ["sequential", "fft", "dwt", "wavelet_packet", "wavelet_packet_max_gauge"]


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
