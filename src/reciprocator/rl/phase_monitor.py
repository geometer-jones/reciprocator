from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..model import ReciprocatorLM


@dataclass(frozen=True)
class PhaseTrajectoryStats:
    token_count: int
    mean_phase_variance: float
    max_phase_variance: float
    mean_phase_delta: float
    phase_delta_variance: float


class PhaseTrajectoryMonitor:
    def record(
        self,
        model: ReciprocatorLM,
        token_ids: Tensor,
        *,
        output_start: int = 0,
    ) -> PhaseTrajectoryStats:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        captured: dict[str, Tensor] = {}

        def capture_hidden(_module, _inputs, output):
            captured["hidden"] = output.detach()

        handle = model.final_norm.register_forward_hook(capture_hidden)
        training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(token_ids)
        finally:
            handle.remove()
            model.train(training)

        hidden = captured.get("hidden")
        if hidden is None:
            return PhaseTrajectoryStats(0, 0.0, 0.0, 0.0, 0.0)
        if not torch.is_complex(hidden):
            raise TypeError("PhaseTrajectoryMonitor expects complex hidden states.")

        # MPS currently lacks torch.angle for complex tensors; these diagnostics
        # are small, so compute phase statistics on CPU without moving the model.
        hidden = hidden.cpu()
        phase = torch.angle(hidden[:, output_start:, :])
        if phase.numel() == 0:
            return PhaseTrajectoryStats(0, 0.0, 0.0, 0.0, 0.0)
        per_token_variance = phase.var(dim=-1, unbiased=False)
        if phase.shape[1] > 1:
            deltas = torch.diff(phase, dim=1)
            mean_delta = deltas.abs().mean()
            delta_variance = deltas.var(unbiased=False)
        else:
            mean_delta = torch.tensor(0.0, device=phase.device)
            delta_variance = torch.tensor(0.0, device=phase.device)
        return PhaseTrajectoryStats(
            token_count=int(phase.shape[1]),
            mean_phase_variance=float(per_token_variance.mean().item()),
            max_phase_variance=float(per_token_variance.max().item()),
            mean_phase_delta=float(mean_delta.item()),
            phase_delta_variance=float(delta_variance.item()),
        )


__all__ = [
    "PhaseTrajectoryMonitor",
    "PhaseTrajectoryStats",
]
