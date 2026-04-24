from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .complex_ops import ComplexLayerNorm, ComplexLinear, ComplexModReLU, canonicalize_normalization_type
from .mixer import ReciprocatorMixer, canonicalize_coupling_type, phase_aware_feature_map


class TokenLift(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        base: float = 10000.0,
        magnitude_type: str = "inverse_frequency_learned",
        phase_type: str = "rope",
        token_phase: str = "semantic",
        token_frequencies: Optional[Tensor] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.magnitude_type = magnitude_type
        self.phase_type = phase_type
        self.token_phase = token_phase
        self.eps = eps
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.token_log_scale_residual = nn.Embedding(vocab_size, 1)
        nn.init.zeros_(self.token_log_scale_residual.weight)

        if token_phase not in {"none", "semantic", "virtual_offset", "semantic_virtual_offset"}:
            raise ValueError(
                "token_phase must be one of {'none', 'semantic', 'virtual_offset', 'semantic_virtual_offset'}."
            )

        if token_phase in {"semantic", "semantic_virtual_offset"}:
            self.phase_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.phase_proj.weight)

        if token_phase in {"virtual_offset", "semantic_virtual_offset"}:
            self.token_alpha = nn.Embedding(vocab_size, 1)
            nn.init.zeros_(self.token_alpha.weight)

        if magnitude_type not in {"learned", "inverse_frequency", "inverse_frequency_learned"}:
            raise ValueError(
                "magnitude_type must be one of {'learned', 'inverse_frequency', 'inverse_frequency_learned'}."
            )

        frequencies = self._resolve_token_frequencies(vocab_size, token_frequencies)
        self.register_buffer("token_frequencies", frequencies, persistent=False)
        self.register_buffer("inverse_frequency_scale", frequencies.mean() / frequencies, persistent=False)

        if phase_type == "rope":
            frequencies = base ** (-torch.arange(hidden_size, dtype=torch.float32) / hidden_size)
            self.register_buffer("omega", frequencies, persistent=False)
        elif phase_type == "locked_wave":
            self.log_omega = nn.Parameter(torch.zeros(()))
            self.log_wavevector = nn.Parameter(torch.zeros(()))
            dims = torch.arange(hidden_size, dtype=torch.float32)
            self.register_buffer("dims", dims, persistent=False)
        elif phase_type == "local_wave":
            num_bands = max(1, int(hidden_size ** 0.5))
            band_size = math.ceil(hidden_size / num_bands)
            self.log_omega_bands = nn.Parameter(torch.zeros(num_bands))
            self.log_wavevector_bands = nn.Parameter(torch.zeros(num_bands))
            dims = torch.arange(hidden_size, dtype=torch.float32)
            self.register_buffer("dims", dims, persistent=False)
            band_id = torch.arange(hidden_size, dtype=torch.long) // band_size
            band_id = band_id.clamp_max(num_bands - 1)
            self.register_buffer("band_id", band_id, persistent=False)
            dim_in_band = (dims - (band_id.float() * band_size)).float()
            self.register_buffer("dim_in_band", dim_in_band, persistent=False)
        else:
            raise ValueError("phase_type must be one of {'rope', 'locked_wave', 'local_wave'}.")

    @staticmethod
    def _resolve_token_frequencies(vocab_size: int, token_frequencies: Optional[Tensor]) -> Tensor:
        if token_frequencies is None:
            ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
            return ranks.reciprocal()

        frequencies = torch.as_tensor(token_frequencies, dtype=torch.float32)
        if frequencies.shape != (vocab_size,):
            raise ValueError("token_frequencies must have shape [vocab_size].")
        if not torch.isfinite(frequencies).all().item():
            raise ValueError("token_frequencies must be finite.")
        if torch.any(frequencies <= 0).item():
            raise ValueError("token_frequencies must be strictly positive.")
        return frequencies

    def _inverse_frequency_amplitude(self, token_ids: Tensor, *, learn_residual: bool) -> Tensor:
        token_profile = F.softplus(self.token_embedding(token_ids))
        profile_norm = torch.sqrt(token_profile.square().sum(dim=-1, keepdim=True).clamp_min(self.eps))
        normalized_profile = token_profile / profile_norm
        scale = self.inverse_frequency_scale[token_ids].unsqueeze(-1)
        if learn_residual:
            scale = scale * torch.exp(self.token_log_scale_residual(token_ids))
        return normalized_profile * scale

    def _compute_phase(
        self, token_ids: Tensor, position_offset: int = 0
    ) -> Tensor:
        if token_ids.ndim < 2:
            raise ValueError("Expected token_ids to have at least shape [batch, seq].")

        batch_size = token_ids.shape[0]
        seq_len = token_ids.shape[1]
        extra_shape = tuple(int(dim) for dim in token_ids.shape[2:])
        device = token_ids.device
        positions = torch.arange(position_offset, position_offset + seq_len, device=device, dtype=torch.float32)

        # Position phase: [seq, D]
        if self.phase_type == "rope":
            pos_phase = positions[:, None] * self.omega[None, :]
        elif self.phase_type == "locked_wave":
            omega = F.softplus(self.log_omega)
            wavevector = self.log_wavevector
            pos_phase = omega * positions[:, None] + wavevector * self.dims[None, :]
        else:
            omega_per_band = F.softplus(self.log_omega_bands)
            kv_per_band = self.log_wavevector_bands
            omega_per_dim = omega_per_band[self.band_id]
            kv_per_dim = kv_per_band[self.band_id]
            pos_phase = omega_per_dim * positions[:, None] + kv_per_dim * self.dim_in_band[None, :]

        # Start from position phase, expand to [batch, seq, D] if token phase is active
        phase_view = (1, seq_len, *([1] * len(extra_shape)), self.hidden_size)
        has_token_phase = self.token_phase != "none"

        if has_token_phase:
            phase = pos_phase.view(phase_view).expand(batch_size, seq_len, *extra_shape, self.hidden_size).clone()
        elif extra_shape:
            return pos_phase.view(phase_view).expand(batch_size, seq_len, *extra_shape, self.hidden_size)
        else:
            return pos_phase  # [seq, D] — no per-token variation

        if self.token_phase in {"semantic", "semantic_virtual_offset"}:
            token_phase_offset = self.phase_proj(self.token_embedding(token_ids))
            phase = phase + token_phase_offset

        if self.token_phase in {"virtual_offset", "semantic_virtual_offset"}:
            alpha = self.token_alpha(token_ids).squeeze(-1)  # [batch, seq]
            if self.phase_type == "rope":
                phase = phase + alpha.unsqueeze(-1) * self.omega.unsqueeze(0)
            elif self.phase_type == "local_wave":
                phase = phase + alpha.unsqueeze(-1) * F.softplus(self.log_omega_bands)[self.band_id].unsqueeze(0)

        return phase

    def forward(self, token_ids: Tensor, position_offset: int = 0) -> Tensor:
        if token_ids.ndim < 2:
            raise ValueError("Expected token_ids to have shape [batch, seq].")

        if self.magnitude_type == "learned":
            token_amplitude = torch.exp(0.1 * torch.tanh(self.token_embedding(token_ids)))
        elif self.magnitude_type == "inverse_frequency":
            token_amplitude = self._inverse_frequency_amplitude(token_ids, learn_residual=False)
        else:
            token_amplitude = self._inverse_frequency_amplitude(token_ids, learn_residual=True)

        phase = self._compute_phase(token_ids, position_offset=position_offset)
        if phase.ndim == 2:
            # [seq, D] → carrier [1, seq, D], broadcasts with amplitude [batch, seq, D]
            carrier = torch.polar(torch.ones_like(phase), phase).unsqueeze(0)
        else:
            # [batch, seq, D] → carrier [batch, seq, D]
            carrier = torch.polar(torch.ones_like(phase), phase)
        return token_amplitude.to(carrier.dtype) * carrier

    def lift_distribution(
        self,
        token_probs: Tensor,
        token_ids: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> Tensor:
        if token_ids is not None:
            if token_probs.shape != token_ids.shape:
                raise ValueError("token_probs and token_ids must have identical shapes for sparse lifting.")
            if token_probs.ndim not in {2, 3}:
                raise ValueError("Expected sparse token_probs/token_ids to have shape [batch, k] or [batch, seq, k].")
            if token_ids.dtype != torch.long:
                token_ids = token_ids.to(torch.long)
            squeeze_seq = token_probs.ndim == 2
            if squeeze_seq:
                token_probs = token_probs.unsqueeze(1)
                token_ids = token_ids.unsqueeze(1)
            lifted = self(token_ids, position_offset=position_offset)
            hidden = (token_probs.to(lifted.dtype).unsqueeze(-1) * lifted).sum(dim=-2)
            return hidden.squeeze(1) if squeeze_seq else hidden

        if token_probs.ndim not in {2, 3}:
            raise ValueError("Expected token_probs to have shape [batch, vocab_size] or [batch, seq, vocab_size].")
        if token_probs.shape[-1] != self.token_embedding.num_embeddings:
            raise ValueError("Expected token_probs to have shape [..., vocab_size].")

        squeeze_seq = token_probs.ndim == 2
        if squeeze_seq:
            token_probs = token_probs.unsqueeze(1)

        batch_size, seq_len, vocab_size = token_probs.shape
        vocab_token_ids = torch.arange(
            vocab_size,
            device=token_probs.device,
            dtype=torch.long,
        ).view(1, 1, vocab_size).expand(batch_size, seq_len, vocab_size)
        lifted_vocab = self(vocab_token_ids, position_offset=position_offset)
        hidden = (token_probs.to(lifted_vocab.dtype).unsqueeze(-1) * lifted_vocab).sum(dim=-2)
        return hidden.squeeze(1) if squeeze_seq else hidden


class ComplexFeedForward(nn.Module):
    def __init__(self, hidden_size: int, expansion_factor: int = 4) -> None:
        super().__init__()
        inner_size = hidden_size * expansion_factor
        self.up_proj = ComplexLinear(hidden_size, inner_size)
        self.activation = ComplexModReLU(inner_size)
        self.down_proj = ComplexLinear(inner_size, hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.up_proj(hidden)))


class ReciprocatorBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        state_shape: Sequence[int],
        ffn_expansion_factor: int = 2,
        enable_self_relation: bool = False,
        dynamic_gains: bool = False,
        gain_projector_rank: int = 8,
        coupling_type: str = "sequential",
        low_frequency_gain: float = 0.5,
        low_frequency_sigma: float = 0.35,
        high_frequency_gain: float = 0.5,
        high_frequency_cutoff: float = 0.5,
        dynamic_spectral_gains: bool = False,
        anisotropic_spectral_gains: bool = False,
        wavelet_levels: Optional[int] = None,
        normalization_type: str = "frobenius",
    ) -> None:
        super().__init__()
        self.mixer = ReciprocatorMixer(
            hidden_size=hidden_size,
            state_shape=state_shape,
            enable_self_relation=enable_self_relation,
            enable_dynamic_gains=dynamic_gains,
            gain_projector_rank=gain_projector_rank,
            coupling_type=coupling_type,
            low_frequency_gain=low_frequency_gain,
            low_frequency_sigma=low_frequency_sigma,
            high_frequency_gain=high_frequency_gain,
            high_frequency_cutoff=high_frequency_cutoff,
            dynamic_spectral_gains=dynamic_spectral_gains,
            anisotropic_spectral_gains=anisotropic_spectral_gains,
            wavelet_levels=wavelet_levels,
            normalization_type=normalization_type,
        )
        self.ffn_norm = ComplexLayerNorm(hidden_size)
        self.ffn = ComplexFeedForward(hidden_size, expansion_factor=ffn_expansion_factor)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.mixer.initial_state(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        hidden: Tensor,
        state: Optional[Tensor] = None,
        *,
        chunk_size: Optional[int] = None,
        track_drift: bool = False,
    ):
        if chunk_size is None and not track_drift:
            delta, next_state = self.mixer(hidden, state)
            hidden = hidden + self.ffn(self.ffn_norm(delta))
            return hidden, next_state

        mixer_output = self.mixer(hidden, state, chunk_size=chunk_size, track_drift=track_drift)
        if track_drift:
            delta, next_state, drift_stats = mixer_output
            hidden = hidden + self.ffn(self.ffn_norm(delta))
            return hidden, next_state, drift_stats

        delta, next_state = mixer_output
        hidden = hidden + self.ffn(self.ffn_norm(delta))
        return hidden, next_state


class MagnitudeReadout(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.output(hidden.abs())


class PhaseAwareReadout(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.output = nn.Linear(3 * hidden_size, vocab_size)
        self.phase_anchor = nn.Parameter(
            torch.complex(torch.ones(hidden_size), torch.zeros(hidden_size))
        )
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        if not torch.is_complex(hidden):
            raise TypeError("PhaseAwareReadout expects a complex-valued tensor.")

        anchor_mag = self.phase_anchor.abs()
        reference = self.phase_anchor / anchor_mag.clamp_min(self.eps)
        reference = reference.unsqueeze(0).unsqueeze(0)

        cross = hidden * reference.conj()
        features = torch.cat([cross.real, cross.imag, hidden.abs()], dim=-1)
        return self.output(features)


class LocalAttentionBlock(nn.Module):
    """Causal sliding-window complex attention block for the hybrid Reciprocator.

    Q, K, V are complex (via ComplexLinear). Attention scores are the
    Hermitian inner product Re(Q K†) / sqrt(head_dim) — real-valued and
    phase-sensitive. Softmax weights are applied to complex V so phase flows
    through to the output. State is a complex KV cache carried across chunks.
    """

    def __init__(self, hidden_size: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        inner = num_heads * self.head_dim
        self.norm = ComplexLayerNorm(hidden_size)
        self.q_proj = ComplexLinear(hidden_size, inner)
        self.k_proj = ComplexLinear(hidden_size, inner)
        self.v_proj = ComplexLinear(hidden_size, inner)
        self.out_proj = ComplexLinear(inner, hidden_size)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        empty = torch.zeros(
            batch_size, 0, self.num_heads, self.head_dim, device=device, dtype=dtype
        )
        return empty, empty

    def forward(
        self, hidden: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch, seq, _ = hidden.shape
        normed = self.norm(hidden)

        Q = self.q_proj(normed).reshape(batch, seq, self.num_heads, self.head_dim)
        K = self.k_proj(normed).reshape(batch, seq, self.num_heads, self.head_dim)
        V = self.v_proj(normed).reshape(batch, seq, self.num_heads, self.head_dim)

        cache_len = 0
        if state is not None:
            K_cache, V_cache = state
            cache_len = K_cache.shape[1]
            if cache_len > 0:
                K = torch.cat([K_cache, K], dim=1)
                V = torch.cat([V_cache, V], dim=1)

        total_kv = K.shape[1]
        if total_kv > self.window_size:
            drop = total_kv - self.window_size
            K = K[:, drop:]
            V = V[:, drop:]
            cache_len = max(0, cache_len - drop)
            total_kv = self.window_size

        # [batch, heads, seq_or_kv, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Hermitian scores: Re(Q K†) / sqrt(head_dim) → real [batch, heads, seq, kv]
        scores = (Q @ K.conj().transpose(-2, -1)).real / math.sqrt(self.head_dim)

        # Causal mask: cache tokens fully visible; causal within current chunk only.
        causal_block = torch.triu(
            torch.full((seq, seq), float("-inf"), dtype=scores.dtype, device=hidden.device),
            diagonal=1,
        )
        if cache_len > 0:
            mask = torch.zeros(seq, total_kv, dtype=scores.dtype, device=hidden.device)
            mask[:, cache_len:] = causal_block
        else:
            mask = causal_block
        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        weights = F.softmax(scores, dim=-1)  # real [batch, heads, seq, kv]

        # Apply real weights to complex V: phase flows through.
        out = torch.complex(weights @ V.real, weights @ V.imag)  # [batch, heads, seq, head_dim]

        out = out.transpose(1, 2).reshape(batch, seq, -1)  # [batch, seq, inner]
        delta = self.out_proj(out)  # complex [batch, seq, hidden]

        new_K = K.transpose(1, 2)[:, -self.window_size :]
        new_V = V.transpose(1, 2)[:, -self.window_size :]
        return hidden + delta, (new_K, new_V)


class ReciprocatorLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        state_shape: Sequence[int],
        num_layers: int = 1,
        ffn_expansion_factor: int = 2,
        readout_type: str = "phase_aware",
        token_magnitude_type: str = "inverse_frequency_learned",
        phase_type: str = "rope",
        token_phase: str = "semantic",
        enable_self_relation: bool = False,
        dynamic_gains: bool = False,
        gain_projector_rank: int = 8,
        enable_cross_layer_state: bool = False,
        coupling_type: str = "sequential",
        low_frequency_gain: float = 0.5,
        low_frequency_sigma: float = 0.35,
        high_frequency_gain: float = 0.5,
        high_frequency_cutoff: float = 0.5,
        dynamic_spectral_gains: bool = False,
        anisotropic_spectral_gains: bool = False,
        wavelet_levels: Optional[int] = None,
        normalization_type: str = "frobenius",
        token_frequencies: Optional[Tensor] = None,
        attention_every_k: int = 0,
        attention_num_heads: int = 8,
        attention_window: int = 256,
        attention_position: str = "after",
        block_layout: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self.M_state = math.prod(self.state_shape)
        self.block_layout = None if block_layout is None else tuple(block_layout)
        self.readout_type = readout_type
        self.enable_self_relation = enable_self_relation
        self.dynamic_gains = dynamic_gains
        self.gain_projector_rank = gain_projector_rank
        self.enable_cross_layer_state = enable_cross_layer_state
        self.coupling_type = canonicalize_coupling_type(coupling_type)
        self.dynamic_spectral_gains = dynamic_spectral_gains
        self.anisotropic_spectral_gains = anisotropic_spectral_gains
        self.normalization_type = canonicalize_normalization_type(normalization_type)

        self.token_lift = TokenLift(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            magnitude_type=token_magnitude_type,
            phase_type=phase_type,
            token_phase=token_phase,
            token_frequencies=token_frequencies,
        )

        if attention_every_k > 0 and attention_position not in {"before", "after"}:
            raise ValueError("attention_position must be 'before' or 'after'.")
        blocks: list[nn.Module] = []
        if self.block_layout is not None:
            for block_kind in self.block_layout:
                if block_kind == "attention":
                    blocks.append(LocalAttentionBlock(hidden_size, attention_num_heads, attention_window))
                    continue
                if block_kind == "reciprocator":
                    blocks.append(
                        ReciprocatorBlock(
                            hidden_size=hidden_size,
                            state_shape=self.state_shape,
                            ffn_expansion_factor=ffn_expansion_factor,
                            enable_self_relation=enable_self_relation,
                            dynamic_gains=dynamic_gains,
                            gain_projector_rank=gain_projector_rank,
                            coupling_type=self.coupling_type,
                            low_frequency_gain=low_frequency_gain,
                            low_frequency_sigma=low_frequency_sigma,
                            high_frequency_gain=high_frequency_gain,
                            high_frequency_cutoff=high_frequency_cutoff,
                            dynamic_spectral_gains=dynamic_spectral_gains,
                            anisotropic_spectral_gains=anisotropic_spectral_gains,
                            wavelet_levels=wavelet_levels,
                            normalization_type=self.normalization_type,
                        )
                    )
                    continue
                raise ValueError("block_layout entries must be 'attention' or 'reciprocator'.")
        else:
            for i in range(num_layers):
                if attention_every_k > 0 and attention_position == "before" and i % attention_every_k == 0:
                    blocks.append(LocalAttentionBlock(hidden_size, attention_num_heads, attention_window))
                blocks.append(
                    ReciprocatorBlock(
                        hidden_size=hidden_size,
                        state_shape=self.state_shape,
                        ffn_expansion_factor=ffn_expansion_factor,
                        enable_self_relation=enable_self_relation,
                        dynamic_gains=dynamic_gains,
                        gain_projector_rank=gain_projector_rank,
                        coupling_type=self.coupling_type,
                        low_frequency_gain=low_frequency_gain,
                        low_frequency_sigma=low_frequency_sigma,
                        high_frequency_gain=high_frequency_gain,
                        high_frequency_cutoff=high_frequency_cutoff,
                        dynamic_spectral_gains=dynamic_spectral_gains,
                        anisotropic_spectral_gains=anisotropic_spectral_gains,
                        wavelet_levels=wavelet_levels,
                        normalization_type=self.normalization_type,
                    )
                )
                if attention_every_k > 0 and attention_position == "after" and (i + 1) % attention_every_k == 0 and i < num_layers - 1:
                    blocks.append(LocalAttentionBlock(hidden_size, attention_num_heads, attention_window))
        self.blocks = nn.ModuleList(blocks)
        self.num_layers = sum(isinstance(block, ReciprocatorBlock) for block in self.blocks)
        cross_layer_feature_size = 3 * self.M_state
        if self.enable_cross_layer_state:
            self.cross_layer_beta = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(max(self.num_layers - 1, 0))]
            )
            self.cross_layer_proj = nn.ModuleList(
                [
                    nn.Linear(cross_layer_feature_size, self.hidden_size, bias=False)
                    for _ in range(max(self.num_layers - 1, 0))
                ]
            )
        else:
            self.cross_layer_beta = nn.ParameterList()
            self.cross_layer_proj = nn.ModuleList()
        self.final_norm = ComplexLayerNorm(hidden_size)
        if readout_type == "magnitude":
            self.readout = MagnitudeReadout(hidden_size=hidden_size, vocab_size=vocab_size)
        elif readout_type == "phase_aware":
            self.readout = PhaseAwareReadout(hidden_size=hidden_size, vocab_size=vocab_size)
        else:
            raise ValueError("readout_type must be one of {'magnitude', 'phase_aware'}.")

    @staticmethod
    def _state_feature_map(state: Tensor) -> Tensor:
        features = phase_aware_feature_map(state, batch_dim=True)
        return features.reshape(state.shape[0], -1)

    @staticmethod
    def _compute_cross_memory_residual(
        kv_cache: Tuple[Tensor, Tensor],
        states: Tuple[Optional[Tensor], ...],
    ) -> float:
        K, V = kv_cache
        if K.numel() == 0 or V.numel() == 0:
            return 0.0

        kv_summary = torch.cat([K, V], dim=-1).mean(dim=(0, 1)).reshape(-1)
        state_summaries = []
        for state in states:
            if state is None or not isinstance(state, Tensor):
                continue
            features = phase_aware_feature_map(state, batch_dim=True)
            state_summaries.append(features.mean(dim=0).reshape(-1))

        if not state_summaries:
            return 0.0

        tensor_summary = torch.cat(state_summaries).to(device=kv_summary.device)
        if torch.is_complex(kv_summary) and not torch.is_complex(tensor_summary):
            tensor_summary = torch.complex(tensor_summary, torch.zeros_like(tensor_summary))
        else:
            tensor_summary = tensor_summary.to(dtype=kv_summary.dtype)

        compare_size = min(kv_summary.numel(), tensor_summary.numel())
        if compare_size == 0:
            return 0.0

        residual = kv_summary[:compare_size] - tensor_summary[:compare_size]
        return float(torch.linalg.vector_norm(residual).item())

    def _inject_cross_layer_state(
        self,
        hidden: Tensor,
        next_state: Tensor,
        *,
        donor_layer_index: int,
    ) -> Tensor:
        if donor_layer_index >= len(self.cross_layer_proj):
            return hidden

        state_features = self._state_feature_map(next_state)
        correction = self.cross_layer_proj[donor_layer_index](state_features)
        correction_c = torch.complex(correction, torch.zeros_like(correction)).unsqueeze(1)
        beta = torch.tanh(self.cross_layer_beta[donor_layer_index]).to(correction.dtype)

        # Note: next_state reflects the full chunk; this correction is non-causal
        # within the chunk but causal across chunks. Consistent with the stale-coupling
        # approximation in _chunked_forward (§13.1).
        return hidden + beta * correction_c

    def _forward_blocks(
        self,
        hidden: Tensor,
        states: Sequence[Optional[Tensor]],
        *,
        chunk_size: Optional[int],
        track_drift: bool,
    ) -> Tuple[Tensor, Tuple[Optional[Tensor], ...], Optional[list[Optional[dict]]]]:
        next_states = []
        block_drift_stats: Optional[list[Optional[dict]]] = [] if track_drift else None
        donor_layer_index = 0

        for block, state in zip(self.blocks, states):
            if isinstance(block, ReciprocatorBlock):
                if chunk_size is None and not track_drift:
                    hidden, next_state = block(hidden, state)
                else:
                    block_output = block(hidden, state, chunk_size=chunk_size, track_drift=track_drift)
                    if track_drift:
                        hidden, next_state, drift_stats = block_output
                        assert block_drift_stats is not None
                        block_drift_stats.append(drift_stats)
                    else:
                        hidden, next_state = block_output

                next_states.append(next_state)
                if self.enable_cross_layer_state:
                    hidden = self._inject_cross_layer_state(
                        hidden,
                        next_state,
                        donor_layer_index=donor_layer_index,
                    )
                donor_layer_index += 1
                continue

            hidden, next_state = block(hidden, state)
            next_states.append(next_state)
            if track_drift:
                assert block_drift_stats is not None
                block_drift_stats.append(None)

        return hidden, tuple(next_states), block_drift_stats

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.cfloat,
    ) -> Tuple[Tensor, ...]:
        return tuple(block.initial_state(batch_size, device=device, dtype=dtype) for block in self.blocks)

    @staticmethod
    def _aggregate_chunk_drift_stats(
        block_drift_stats: Sequence[Optional[dict]],
        *,
        chunk_size: Optional[int],
    ) -> Optional[dict]:
        filtered = [stats for stats in block_drift_stats if stats is not None]
        if not filtered:
            return None
        return {
            "mean_drift": float(sum(stats["mean_drift"] for stats in filtered) / len(filtered)),
            "max_drift": float(max(stats["max_drift"] for stats in filtered)),
            "K": filtered[0]["K"] if chunk_size is None else int(chunk_size),
            "block_count": len(filtered),
        }

    def forward(
        self,
        token_ids: Tensor,
        *,
        states: Optional[Sequence[Optional[Tensor]]] = None,
        position_offset: int = 0,
        chunk_size: Optional[int] = None,
        track_drift: bool = False,
    ):
        hidden = self.token_lift(token_ids, position_offset=position_offset)

        if states is None:
            states = (None,) * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("Number of states must match the number of blocks.")

        hidden, next_states, block_drift_stats = self._forward_blocks(
            hidden,
            states,
            chunk_size=chunk_size,
            track_drift=track_drift,
        )

        logits = self.readout(self.final_norm(hidden))
        if track_drift:
            assert block_drift_stats is not None
            drift_stats = self._aggregate_chunk_drift_stats(block_drift_stats, chunk_size=chunk_size)
            return logits, tuple(next_states), drift_stats
        return logits, tuple(next_states)

    def loss(
        self,
        token_ids: Tensor,
        targets: Tensor,
        *,
        states: Optional[Sequence[Optional[Tensor]]] = None,
        position_offset: int = 0,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        logits, next_states = self(
            token_ids,
            states=states,
            position_offset=position_offset,
        )
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return loss, next_states
