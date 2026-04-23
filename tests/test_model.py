import torch

from reciprocator.model import PhaseAwareReadout, ReciprocatorLM, TokenLift


def random_complex(*shape: int) -> torch.Tensor:
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


def next_token_targets(token_ids: torch.Tensor) -> torch.Tensor:
    targets = torch.zeros_like(token_ids)
    targets[:, :-1] = token_ids[:, 1:]
    return targets


def test_token_lift_returns_complex_hidden_stream() -> None:
    lift = TokenLift(vocab_size=11, hidden_size=8)
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])

    hidden = lift(token_ids)

    assert hidden.shape == (2, 3, 8)
    assert torch.is_complex(hidden)
    assert torch.all(hidden.abs() > 0)


def test_token_lift_supports_inverse_frequency_scaling() -> None:
    frequencies = torch.tensor([4.0, 2.0, 1.0])
    lift = TokenLift(
        vocab_size=3,
        hidden_size=4,
        magnitude_type="inverse_frequency",
        token_frequencies=frequencies,
    )
    with torch.no_grad():
        lift.token_embedding.weight.zero_()

    hidden = lift(torch.tensor([[0, 1, 2]]), position_offset=0)
    norms = torch.sqrt(hidden.abs().square().sum(dim=-1)).squeeze(0)
    expected = frequencies.mean() / frequencies

    assert torch.allclose(norms, expected, atol=1e-5)


def test_token_lift_inverse_frequency_learned_starts_from_inverse_frequency() -> None:
    frequencies = torch.tensor([4.0, 2.0, 1.0])
    inverse_frequency = TokenLift(
        vocab_size=3,
        hidden_size=4,
        magnitude_type="inverse_frequency",
        token_frequencies=frequencies,
    )
    inverse_frequency_learned = TokenLift(
        vocab_size=3,
        hidden_size=4,
        magnitude_type="inverse_frequency_learned",
        token_frequencies=frequencies,
    )
    with torch.no_grad():
        inverse_frequency.token_embedding.weight.fill_(0.25)
        inverse_frequency_learned.token_embedding.weight.copy_(inverse_frequency.token_embedding.weight)

    token_ids = torch.tensor([[0, 1, 2]])
    inverse_hidden = inverse_frequency(token_ids)
    learned_hidden = inverse_frequency_learned(token_ids)

    assert torch.allclose(learned_hidden, inverse_hidden, atol=1e-6)


def test_token_lift_supports_probability_predictions() -> None:
    lift = TokenLift(vocab_size=5, hidden_size=4)
    token_probs = torch.softmax(torch.randn(3, 5), dim=-1)

    hidden = lift.lift_distribution(token_probs, position_offset=2)

    assert hidden.shape == (3, 4)
    assert torch.is_complex(hidden)


def test_token_lift_supports_sparse_probability_predictions() -> None:
    lift = TokenLift(vocab_size=5, hidden_size=4)
    token_probs = torch.softmax(torch.randn(2, 3), dim=-1)
    token_ids = torch.tensor([[0, 2, 4], [1, 3, 0]])

    hidden = lift.lift_distribution(token_probs, token_ids, position_offset=2)

    assert hidden.shape == (2, 4)
    assert torch.is_complex(hidden)


def test_lm_forward_returns_logits_and_block_states() -> None:
    model = ReciprocatorLM(vocab_size=13, hidden_size=8, state_shape=(2, 3), num_layers=2)
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits, states = model(token_ids)

    assert logits.shape == (2, 4, 13)
    assert len(states) == 2
    assert states[0].shape == (2, 2, 3)
    assert isinstance(model.readout, PhaseAwareReadout)
    assert model.coupling_type == "sequential"
    assert model.token_lift.magnitude_type == "inverse_frequency_learned"
    assert model.token_lift.token_phase == "semantic"


def test_phase_aware_readout_preserves_output_shape_and_anchor_direction() -> None:
    readout = PhaseAwareReadout(hidden_size=5, vocab_size=7)
    hidden = random_complex(2, 3, 5)
    with torch.no_grad():
        readout.phase_anchor.copy_(torch.complex(torch.ones(5), torch.zeros(5)))
    logits = readout(hidden)

    with torch.no_grad():
        readout.phase_anchor.copy_(torch.complex(7.0 * torch.ones(5), torch.zeros(5)))
    rescaled_anchor_logits = readout(hidden)

    assert logits.shape == (2, 3, 7)
    assert readout.output.in_features == 15
    assert torch.allclose(rescaled_anchor_logits, logits, atol=1e-5)


def test_phase_aware_readout_supports_legacy_state_dict_with_strict_false() -> None:
    readout = PhaseAwareReadout(hidden_size=5, vocab_size=7)
    legacy_state_dict = {
        "output.weight": readout.output.weight.detach().clone(),
        "output.bias": readout.output.bias.detach().clone(),
    }

    incompatible_keys = readout.load_state_dict(legacy_state_dict, strict=False)

    assert incompatible_keys.missing_keys == ["phase_anchor"]
    assert incompatible_keys.unexpected_keys == []


def test_lm_supports_phase_aware_readout() -> None:
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=2,
        readout_type="phase_aware",
    )
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits, states = model(token_ids)

    assert logits.shape == (2, 4, 13)
    assert len(states) == 2
    assert isinstance(model.readout, PhaseAwareReadout)


def test_lm_returns_chunk_drift_stats_when_requested() -> None:
    model = ReciprocatorLM(vocab_size=13, hidden_size=8, state_shape=(2, 3), num_layers=2)
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits, states, drift_stats = model(token_ids, chunk_size=2, track_drift=True)

    assert logits.shape == (2, 4, 13)
    assert len(states) == 2
    assert drift_stats is not None
    assert drift_stats["K"] == 2
    assert drift_stats["mean_drift"] >= 0.0
    assert drift_stats["max_drift"] >= drift_stats["mean_drift"]


def test_lm_supports_inverse_frequency_token_lift() -> None:
    frequencies = torch.linspace(1.0, 13.0, steps=13).flip(0)
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=1,
        token_magnitude_type="inverse_frequency",
        token_frequencies=frequencies,
    )
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits, states = model(token_ids)

    assert logits.shape == (2, 4, 13)
    assert len(states) == 1


def test_lm_supports_inverse_frequency_learned_token_lift() -> None:
    frequencies = torch.linspace(1.0, 13.0, steps=13).flip(0)
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=1,
        token_magnitude_type="inverse_frequency_learned",
        token_frequencies=frequencies,
    )
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    logits, states = model(token_ids)

    assert logits.shape == (2, 4, 13)
    assert len(states) == 1


def test_lm_plumbs_self_relation_flag_to_blocks() -> None:
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=2,
        enable_self_relation=True,
    )

    assert model.enable_self_relation is True
    assert all(block.mixer.enable_self_relation for block in model.blocks)


def test_lm_plumbs_anticipator_relation_flag() -> None:
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=2,
        enable_anticipator_relation=True,
    )

    assert model.enable_anticipator_relation is True
    assert model.anticipator_relation_logit is not None


def test_lm_plumbs_cross_layer_state_flag() -> None:
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=2,
        enable_cross_layer_state=True,
    )

    assert model.enable_cross_layer_state is True
    assert len(model.cross_layer_proj) == 1
    assert len(model.cross_layer_beta) == 1


def test_lm_plumbs_normalization_flag_to_blocks() -> None:
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=2,
        normalization_type="per_mode",
    )

    assert model.normalization_type == "per_mode"
    assert all(block.mixer.normalization_type == "per_mode" for block in model.blocks)


def test_lm_plumbs_coupling_type_to_blocks() -> None:
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=2,
        coupling_type="wavelet_packet_max_gauge",
    )

    assert model.coupling_type == "wavelet_packet_max_gauge"
    assert all(block.mixer.coupling_type == "wavelet_packet_max_gauge" for block in model.blocks)


def test_lm_plumbs_dynamic_spectral_gains_to_spectral_blocks() -> None:
    model = ReciprocatorLM(
        vocab_size=13,
        hidden_size=8,
        state_shape=(2, 3),
        num_layers=2,
        coupling_type="fft",
        dynamic_spectral_gains=True,
        anisotropic_spectral_gains=True,
        gain_projector_rank=4,
    )

    assert model.dynamic_spectral_gains is True
    assert model.anisotropic_spectral_gains is True
    assert all(block.mixer.dynamic_spectral_gains for block in model.blocks)
    assert all(block.mixer.anisotropic_spectral_gains for block in model.blocks)
    assert all(block.mixer.coupling.dynamic_spectral_gains for block in model.blocks)
    assert all(block.mixer.coupling.anisotropic_spectral_gains for block in model.blocks)


def test_invalid_readout_type_raises() -> None:
    try:
        ReciprocatorLM(vocab_size=13, hidden_size=8, state_shape=(2, 3), readout_type="invalid")
    except ValueError as error:
        assert "readout_type" in str(error)
    else:
        raise AssertionError("Expected invalid readout_type to raise ValueError.")


def test_invalid_token_magnitude_type_raises() -> None:
    try:
        TokenLift(vocab_size=13, hidden_size=8, magnitude_type="invalid")
    except ValueError as error:
        assert "magnitude_type" in str(error)
    else:
        raise AssertionError("Expected invalid magnitude_type to raise ValueError.")


def test_invalid_coupling_type_raises() -> None:
    try:
        ReciprocatorLM(vocab_size=13, hidden_size=8, state_shape=(2, 3), coupling_type="invalid")
    except ValueError as error:
        assert "coupling_type" in str(error)
    else:
        raise AssertionError("Expected invalid coupling_type to raise ValueError.")


def test_lm_streaming_matches_full_sequence() -> None:
    model = ReciprocatorLM(vocab_size=9, hidden_size=6, state_shape=(2, 2), num_layers=1)
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    full_logits, full_states = model(token_ids)

    states = model.initial_state(token_ids.shape[0], device=token_ids.device)
    step_logits = []
    for position in range(token_ids.shape[1]):
        logits_t, states = model(
            token_ids[:, position : position + 1],
            states=states,
            position_offset=position,
        )
        step_logits.append(logits_t)

    streamed_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(streamed_logits, full_logits, atol=1e-5)
    assert torch.allclose(states[0], full_states[0], atol=1e-5)


def test_lm_anticipator_relation_is_inert_at_initialization() -> None:
    torch.manual_seed(0)
    model = ReciprocatorLM(vocab_size=9, hidden_size=6, state_shape=(2, 2), num_layers=1)
    anticipatory_model = ReciprocatorLM(
        vocab_size=9,
        hidden_size=6,
        state_shape=(2, 2),
        num_layers=1,
        enable_anticipator_relation=True,
    )
    incompatible_keys = anticipatory_model.load_state_dict(model.state_dict(), strict=False)

    assert incompatible_keys.missing_keys == ["anticipator_relation_logit"]
    assert incompatible_keys.unexpected_keys == []

    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = next_token_targets(token_ids)
    logits, states = model(token_ids)
    anticipatory_logits, anticipatory_states, next_ant = anticipatory_model(token_ids, targets=targets)

    assert torch.allclose(anticipatory_logits, logits, atol=1e-5)
    assert torch.allclose(anticipatory_states[0], states[0], atol=1e-5)
    assert next_ant is None


def test_lm_anticipator_relation_changes_outputs_when_gain_is_nonzero() -> None:
    torch.manual_seed(0)
    model = ReciprocatorLM(vocab_size=9, hidden_size=6, state_shape=(2, 2), num_layers=1)
    anticipatory_model = ReciprocatorLM(
        vocab_size=9,
        hidden_size=6,
        state_shape=(2, 2),
        num_layers=1,
        enable_anticipator_relation=True,
    )
    anticipatory_model.load_state_dict(model.state_dict(), strict=False)

    with torch.no_grad():
        anticipatory_model.anticipator_relation_logit.fill_(2.0)

    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = next_token_targets(token_ids)
    logits, states = model(token_ids)
    anticipatory_logits, anticipatory_states, _ = anticipatory_model(token_ids, targets=targets)

    assert not torch.allclose(anticipatory_logits, logits, atol=1e-5)
    assert not torch.allclose(anticipatory_states[0], states[0], atol=1e-5)


def test_lm_anticipator_relation_streaming_matches_full_sequence() -> None:
    model = ReciprocatorLM(
        vocab_size=9,
        hidden_size=6,
        state_shape=(2, 2),
        num_layers=1,
        enable_anticipator_relation=True,
    )
    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = next_token_targets(token_ids)

    full_logits, full_states, full_next_ant = model(token_ids, targets=targets)

    states = model.initial_state(token_ids.shape[0], device=token_ids.device)
    step_logits = []
    for position in range(token_ids.shape[1]):
        step_targets = targets[:, position : position + 1]
        logits_t, states, step_next_ant = model(
            token_ids[:, position : position + 1],
            states=states,
            position_offset=position,
            targets=step_targets,
        )
        step_logits.append(logits_t)
        assert step_next_ant is None

    streamed_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(streamed_logits, full_logits, atol=1e-5)
    assert torch.allclose(states[0], full_states[0], atol=1e-5)
    assert full_next_ant is None


def test_lm_anticipator_relation_inference_returns_next_ant() -> None:
    model = ReciprocatorLM(
        vocab_size=9,
        hidden_size=6,
        state_shape=(2, 2),
        num_layers=1,
        enable_anticipator_relation=True,
    )
    token_ids = torch.tensor([[1, 2, 3], [4, 3, 2]])

    logits, states, next_ant = model(token_ids)

    assert logits.shape == (2, 3, 9)
    assert len(states) == 1
    assert next_ant is not None
    assert next_ant.shape == (2, 3, 6)


def test_lm_cross_layer_state_is_inert_at_initialization() -> None:
    torch.manual_seed(0)
    model = ReciprocatorLM(vocab_size=9, hidden_size=6, state_shape=(2, 2), num_layers=2)
    cross_layer_model = ReciprocatorLM(
        vocab_size=9,
        hidden_size=6,
        state_shape=(2, 2),
        num_layers=2,
        enable_cross_layer_state=True,
    )
    incompatible_keys = cross_layer_model.load_state_dict(model.state_dict(), strict=False)

    assert sorted(incompatible_keys.missing_keys) == ["cross_layer_beta.0", "cross_layer_proj.0.weight"]
    assert incompatible_keys.unexpected_keys == []

    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    logits, states = model(token_ids)
    cross_layer_logits, cross_layer_states = cross_layer_model(token_ids)

    assert torch.allclose(cross_layer_logits, logits, atol=1e-5)
    assert all(torch.allclose(cross, base, atol=1e-5) for cross, base in zip(cross_layer_states, states))


def test_lm_cross_layer_state_changes_outputs_when_gate_is_nonzero() -> None:
    torch.manual_seed(0)
    model = ReciprocatorLM(vocab_size=9, hidden_size=6, state_shape=(2, 2), num_layers=2)
    cross_layer_model = ReciprocatorLM(
        vocab_size=9,
        hidden_size=6,
        state_shape=(2, 2),
        num_layers=2,
        enable_cross_layer_state=True,
    )
    cross_layer_model.load_state_dict(model.state_dict(), strict=False)

    with torch.no_grad():
        cross_layer_model.cross_layer_beta[0].fill_(2.0)
        cross_layer_model.cross_layer_proj[0].weight.normal_(mean=0.0, std=0.05)

    token_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    logits, states = model(token_ids)
    cross_layer_logits, cross_layer_states = cross_layer_model(token_ids)

    assert not torch.allclose(cross_layer_logits, logits, atol=1e-5)
    assert torch.allclose(cross_layer_states[0], states[0], atol=1e-5)
    assert not torch.allclose(cross_layer_states[1], states[1], atol=1e-5)


def test_lm_is_prefix_causal() -> None:
    model = ReciprocatorLM(vocab_size=9, hidden_size=6, state_shape=(2, 2), num_layers=1)
    prefix = torch.tensor([[1, 2, 3], [3, 2, 1]])
    suffix_a = torch.tensor([[4, 5], [5, 4]])
    suffix_b = torch.tensor([[6, 7], [7, 6]])

    logits_a, _ = model(torch.cat([prefix, suffix_a], dim=1))
    logits_b, _ = model(torch.cat([prefix, suffix_b], dim=1))

    assert torch.allclose(logits_a[:, : prefix.shape[1]], logits_b[:, : prefix.shape[1]], atol=1e-5)


def test_lm_training_smoke_reduces_loss() -> None:
    torch.manual_seed(0)

    model = ReciprocatorLM(vocab_size=5, hidden_size=12, state_shape=(2, 3), num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    token_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, 2],
            [1, 2, 3, 4, 0, 1, 2, 3],
            [2, 3, 4, 0, 1, 2, 3, 4],
            [3, 4, 0, 1, 2, 3, 4, 0],
        ]
    )
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = model.loss(inputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] < losses[0]
