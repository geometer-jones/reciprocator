import torch

from reciprocator.rl.phase_monitor import PhaseTrajectoryMonitor
from reciprocator.rl.training import build_lisp_tokenizer
from reciprocator.model import ReciprocatorLM


def test_phase_monitor_records_hidden_phase_statistics() -> None:
    tokenizer = build_lisp_tokenizer()
    model = ReciprocatorLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=8,
        state_shape=(2, 2),
        num_layers=1,
        token_frequencies=torch.ones(tokenizer.vocab_size),
    )
    token_ids = tokenizer.encode("(+ 1 2)\n3")

    stats = PhaseTrajectoryMonitor().record(model, token_ids, output_start=8)

    assert stats.token_count > 0
    assert stats.mean_phase_variance >= 0.0
    assert stats.phase_delta_variance >= 0.0
