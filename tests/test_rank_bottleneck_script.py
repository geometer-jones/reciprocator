import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from reciprocator.model import ReciprocatorLM


def load_rank_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_rank_bottleneck.py"
    spec = importlib.util.spec_from_file_location("rank_bottleneck_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_effective_rank_detects_rank_one_matrix() -> None:
    script = load_rank_script()
    matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=torch.float32,
    )

    metrics = script.spectrum_metrics(matrix)

    assert metrics.effective_rank == pytest.approx(1.0)
    assert metrics.stable_rank == pytest.approx(1.0)
    assert metrics.top_singular_mass == pytest.approx(1.0)


def test_compare_rank_reports_recovery_after_coupling() -> None:
    script = load_rank_script()
    relational = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.cfloat)
    coupled = torch.eye(2, dtype=torch.cfloat).unsqueeze(0)

    observations = script.compare_rank(
        relational,
        coupled,
        layer=0,
        timestep=0,
        modes=(0,),
        rank1_erank_threshold=1.25,
        rank1_top_mass_threshold=0.9,
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.hadamard_near_rank1 is True
    assert observation.hadamard.effective_rank == 1.0
    assert observation.coupled.effective_rank == 2.0
    assert observation.recovered_rank == 1.0


def test_trace_rank_observations_replays_model_forward() -> None:
    script = load_rank_script()
    torch.manual_seed(0)
    model = ReciprocatorLM(vocab_size=7, hidden_size=6, state_shape=(2, 2), num_layers=2)
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])

    expected_logits, expected_states = model(token_ids)
    traced_logits, traced_states, observations = script.trace_rank_observations(
        model,
        token_ids,
        modes=None,
        rank1_erank_threshold=1.25,
        rank1_top_mass_threshold=0.9,
    )

    assert torch.allclose(traced_logits, expected_logits, atol=1e-6)
    assert len(traced_states) == len(expected_states)
    for traced_state, expected_state in zip(traced_states, expected_states):
        assert torch.allclose(traced_state, expected_state, atol=1e-6)
    assert len(observations) == model.num_layers * token_ids.shape[1] * len(model.state_shape)
