"""Shared utilities for Lisp RL experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reciprocator.model import ReciprocatorLM  # noqa: E402
from reciprocator.rl.problem_gen import ProblemGenerator, default_difficulty_for_stage  # noqa: E402
from reciprocator.training import (  # noqa: E402
    CharTokenizer,
    _compute_layer_mode_residual_norms,
    _compute_mode_pruning_residual_norms,
    _compute_mode_residual_norms,
    _compute_mode_slice_activation_variances,
    _update_ema,
    _update_tensor_list_ema,
)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict: {path}")
    return payload


def tokenizer_from_checkpoint(payload: Mapping[str, Any]) -> CharTokenizer:
    vocabulary = payload.get("vocabulary")
    if not isinstance(vocabulary, list) or not all(isinstance(item, str) for item in vocabulary):
        raise ValueError("Checkpoint is missing a character vocabulary.")
    return CharTokenizer(stoi={character: index for index, character in enumerate(vocabulary)}, itos=tuple(vocabulary))


def model_from_checkpoint(payload: Mapping[str, Any], tokenizer: CharTokenizer) -> tuple[ReciprocatorLM, dict[str, Any]]:
    lm_config = payload.get("config")
    if not isinstance(lm_config, dict):
        raise ValueError("Checkpoint is missing a config dict.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")

    model = ReciprocatorLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(lm_config["hidden_size"]),
        state_shape=tuple(lm_config["state_shape"]),
        num_layers=int(lm_config["num_layers"]),
        ffn_expansion_factor=int(lm_config.get("ffn_expansion_factor", 2)),
        readout_type=str(lm_config.get("readout_type", "phase_aware")),
        token_magnitude_type=str(lm_config.get("token_magnitude_type", "inverse_frequency_learned")),
        phase_type=str(lm_config.get("phase_type", "rope")),
        token_phase=str(lm_config.get("token_phase", "semantic")),
        enable_self_relation=bool(lm_config.get("enable_self_relation", False)),
        dynamic_gains=bool(lm_config.get("dynamic_gains", False)),
        gain_projector_rank=int(lm_config.get("gain_projector_rank", 8)),
        enable_cross_layer_state=bool(lm_config.get("enable_cross_layer_state", False)),
        coupling_type=str(lm_config.get("coupling_type", "sequential")),
        low_frequency_gain=float(lm_config.get("low_frequency_gain", 0.5)),
        low_frequency_sigma=float(lm_config.get("low_frequency_sigma", 0.35)),
        high_frequency_gain=float(lm_config.get("high_frequency_gain", 0.5)),
        high_frequency_cutoff=float(lm_config.get("high_frequency_cutoff", 0.5)),
        dynamic_spectral_gains=bool(lm_config.get("dynamic_spectral_gains", False)),
        anisotropic_spectral_gains=bool(lm_config.get("anisotropic_spectral_gains", False)),
        wavelet_levels=lm_config.get("wavelet_levels"),
        normalization_type=str(lm_config.get("normalization_type", "frobenius")),
        token_frequencies=torch.ones(tokenizer.vocab_size),
        attention_every_k=int(lm_config.get("attention_every_k", 0)),
        attention_num_heads=int(lm_config.get("attention_num_heads", 8)),
        attention_window=int(lm_config.get("attention_window", 256)),
        attention_position=str(lm_config.get("attention_position", "after")),
    )
    model.load_state_dict(state_dict)
    return model, lm_config


@dataclass
class ResidualDiagnosticsTracker:
    tokenizer: CharTokenizer
    stage: int
    batch_size: int = 8
    seq_len: int = 64
    seed: int = 0
    ema_decay: float = 0.8
    growth_residual_threshold: float = 0.12
    residual_saturate_threshold: float = 0.07
    prune_threshold: float = 0.4

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.seq_len <= 0:
            raise ValueError("residual diagnostic batch size and sequence length must be positive.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("residual EMA decay must be in [0, 1).")
        self._mode_residual_ema: Optional[torch.Tensor] = None
        self._layer_mode_residual_ema: Optional[torch.Tensor] = None
        self._mode_redundancy_ema: Optional[torch.Tensor] = None
        self._mode_slice_activation_variance_ema: Optional[list[torch.Tensor]] = None

    def _diagnostic_tokens(self, *, step: int, device: torch.device) -> torch.Tensor:
        rng = random.Random(self.seed + step * 1009)
        generator = ProblemGenerator(rng)
        difficulty = default_difficulty_for_stage(self.stage)
        pad_id = self.tokenizer.stoi.get(" ", 0)
        rows = []
        for _ in range(self.batch_size):
            problem = generator.generate_problem(difficulty)
            text = f"{problem.prompt_expression}\n{problem.expected_result_text}\n"
            encoded = self.tokenizer.encode(text[: self.seq_len])
            if encoded.numel() < self.seq_len:
                padding = torch.full((self.seq_len - encoded.numel(),), pad_id, dtype=torch.long)
                encoded = torch.cat([encoded, padding], dim=0)
            rows.append(encoded[: self.seq_len])
        return torch.stack(rows, dim=0).to(device)

    def record(self, model: ReciprocatorLM, *, step: int) -> dict[str, object]:
        device = next(model.parameters()).device
        token_ids = self._diagnostic_tokens(step=step, device=device)
        was_training = model.training
        try:
            mode_residual_norms = _compute_mode_residual_norms(model, token_ids)
            layer_mode_residual_norms = _compute_layer_mode_residual_norms(model, token_ids)
            mode_redundancy_norms = _compute_mode_pruning_residual_norms(model, token_ids)
            mode_slice_variances = _compute_mode_slice_activation_variances(model, token_ids)
        finally:
            model.train(was_training)

        self._mode_residual_ema = _update_ema(self._mode_residual_ema, mode_residual_norms, self.ema_decay)
        self._layer_mode_residual_ema = _update_ema(
            self._layer_mode_residual_ema,
            layer_mode_residual_norms,
            self.ema_decay,
        )
        self._mode_redundancy_ema = _update_ema(self._mode_redundancy_ema, mode_redundancy_norms, self.ema_decay)
        self._mode_slice_activation_variance_ema = _update_tensor_list_ema(
            self._mode_slice_activation_variance_ema,
            mode_slice_variances,
            self.ema_decay,
        )

        max_growth_pressure = float((self._mode_residual_ema / self.growth_residual_threshold).max().item())
        pruning_pressure = self.prune_threshold / self._mode_redundancy_ema.clamp_min(1e-12)
        prune_candidate_modes = [
            mode
            for mode, residual in enumerate(self._mode_redundancy_ema.tolist())
            if float(residual) <= self.prune_threshold
        ]
        rank_saturated = bool(torch.all(self._mode_residual_ema <= self.residual_saturate_threshold).item())
        return {
            "step": step,
            "state_shape": list(model.state_shape),
            "mode_residual_norms": mode_residual_norms.tolist(),
            "mode_residual_ema": self._mode_residual_ema.tolist(),
            "layer_mode_residual_norms": layer_mode_residual_norms.tolist(),
            "layer_mode_residual_ema": self._layer_mode_residual_ema.tolist(),
            "mode_redundancy_norms": mode_redundancy_norms.tolist(),
            "mode_redundancy_ema": self._mode_redundancy_ema.tolist(),
            "prune_threshold": self.prune_threshold,
            "mode_pruning_pressure": pruning_pressure.tolist(),
            "max_pruning_pressure": float(pruning_pressure.max().item()),
            "prune_candidate_modes": prune_candidate_modes,
            "mode_slice_activation_variance_ema": [
                variances.tolist() for variances in self._mode_slice_activation_variance_ema
            ],
            "growth_residual_threshold": self.growth_residual_threshold,
            "residual_saturate_threshold": self.residual_saturate_threshold,
            "max_growth_pressure": max_growth_pressure,
            "rank_residuals_saturated": rank_saturated,
        }
