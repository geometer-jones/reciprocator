from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .complex_ops import (
    ComplexLinear,
    ComplexLayerNorm,
    RealToComplexLinear,
    canonicalize_normalization_type,
    normalize_complex,
)


def _flatten_complex_features(x: Tensor) -> Tensor:
    return torch.cat([x.real, x.imag], dim=-1)


def phase_aware_feature_map(x: Tensor, *, batch_dim: bool) -> Tensor:
    if not torch.is_complex(x):
        raise TypeError("phase_aware_feature_map expects a complex-valued tensor.")
    if x.ndim < (2 if batch_dim else 1):
        raise ValueError("phase_aware_feature_map received an input with too few dimensions.")

    feature_axis = 1 if batch_dim else 0
    reference_dims = tuple(range(1, x.ndim)) if batch_dim else tuple(range(x.ndim))
    reference = x.mean(dim=reference_dims, keepdim=True)
    cross_product = x * reference.conj()
    return torch.stack([cross_product.real, cross_product.imag, x.abs()], dim=feature_axis)


def canonicalize_coupling_type(coupling_type: str) -> str:
    normalized = coupling_type.strip().lower().replace("-", "_")
    aliases = {
        "sequential": "sequential",
        "fft": "fft",
        "fourier": "fft",
        "dwt": "dwt",
        "wavelet_packet": "wavelet_packet",
        "wavelet_packet_entropy": "wavelet_packet",
        "wavelet_packet_phase": "wavelet_packet_max_gauge",
        "wavelet_packet_max_gauge": "wavelet_packet_max_gauge",
    }
    if normalized not in aliases:
        raise ValueError(
            "coupling_type must be one of "
            "{'sequential', 'fft', 'dwt', 'wavelet_packet', 'wavelet_packet_max_gauge'}."
        )
    return aliases[normalized]


class TensorSignalProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        state_shape: Sequence[int],
        phase_scale: float = math.pi,
        normalization_type: str = "frobenius",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self.state_size = math.prod(self.state_shape)
        self.phase_scale = phase_scale
        self.normalization_type = canonicalize_normalization_type(normalization_type)
        self.eps = eps

        self.magnitude_proj = nn.Linear(2 * hidden_size, self.state_size)
        self.phase_proj = nn.Linear(2 * hidden_size, self.state_size)

    def forward(self, hidden: Tensor) -> Tensor:
        features = _flatten_complex_features(hidden)
        magnitude = F.softplus(self.magnitude_proj(features))
        phase = self.phase_scale * torch.tanh(self.phase_proj(features))
        signal = torch.polar(magnitude, phase).reshape(hidden.shape[0], *self.state_shape)
        return normalize_complex(
            signal,
            normalization_type=self.normalization_type,
            dims=tuple(range(1, signal.ndim)),
            eps=self.eps,
        )


class SequentialModeCoupling(nn.Module):
    def __init__(self, state_shape: Sequence[int], tau_init: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self.rank = len(self.state_shape)
        self.state_size = math.prod(self.state_shape)
        self.eps = eps

        self.mode_weights = nn.ParameterList(
            [
                nn.Parameter(0.02 * torch.randn(dim, dim, dtype=torch.cfloat))
                for dim in self.state_shape
            ]
        )
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))

    def _mode_permutation(self, mode: int, ndim: int) -> Tuple[int, ...]:
        state_axis = mode + 1
        return (0, state_axis, *[axis for axis in range(1, ndim) if axis != state_axis])

    def _unfold_mode(self, x: Tensor, mode: int) -> Tensor:
        permutation = self._mode_permutation(mode, x.ndim)
        permuted = x.permute(permutation)
        return permuted.reshape(x.shape[0], x.shape[mode + 1], -1)

    def _fold_mode(self, unfolded: Tensor, original_shape: Tuple[int, ...], mode: int) -> Tensor:
        permutation = self._mode_permutation(mode, len(original_shape))
        permuted_shape = (
            original_shape[0],
            original_shape[mode + 1],
            *[original_shape[axis] for axis in range(1, len(original_shape)) if axis != mode + 1],
        )
        permuted = unfolded.reshape(permuted_shape)
        inverse = [0] * len(permutation)
        for index, axis in enumerate(permutation):
            inverse[axis] = index
        return permuted.permute(tuple(inverse))

    def _mode_matrix(self, unfolded: Tensor, mode: int) -> Tensor:
        weight = self.mode_weights[mode]
        left = torch.einsum("ij,bjk->bik", weight, unfolded)
        score = torch.matmul(left, unfolded.transpose(-1, -2))

        tau = torch.exp(self.log_tau).clamp_min(self.eps)
        scale = tau * math.sqrt(self.state_size / self.state_shape[mode])
        magnitude = score.abs()
        routing = torch.softmax(magnitude / scale, dim=-1)

        safe_magnitude = magnitude.clamp_min(self.eps)
        phase = score / safe_magnitude
        phase = torch.where(magnitude > self.eps, phase, torch.ones_like(phase))
        return routing.to(score.dtype) * phase

    def _apply_mode_matrix(self, x: Tensor, mode: int, matrix: Tensor) -> Tensor:
        unfolded = self._unfold_mode(x, mode)
        mixed = torch.matmul(matrix, unfolded)
        return self._fold_mode(mixed, tuple(x.shape), mode)

    def forward(self, tensor: Tensor) -> Tensor:
        mixed = tensor
        for mode in range(self.rank):
            mode_matrix = self._mode_matrix(self._unfold_mode(mixed, mode), mode)
            mixed = self._apply_mode_matrix(mixed, mode, mode_matrix)
        return mixed


class _SpectralCouplingBase(nn.Module):
    def __init__(
        self,
        state_shape: Sequence[int],
        *,
        low_frequency_gain: float,
        low_frequency_sigma: float,
        high_frequency_gain: float,
        high_frequency_cutoff: float,
        dynamic_spectral_gains: bool = False,
        anisotropic_spectral_gains: bool = False,
        gain_projector_rank: int = 8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self.state_size = math.prod(self.state_shape)
        self.low_frequency_gain = float(low_frequency_gain)
        self.low_frequency_sigma = float(low_frequency_sigma)
        self.high_frequency_gain = float(high_frequency_gain)
        self.high_frequency_cutoff = float(high_frequency_cutoff)
        self.dynamic_spectral_gains = dynamic_spectral_gains
        self.anisotropic_spectral_gains = anisotropic_spectral_gains
        self.gain_projector_rank = gain_projector_rank
        self.eps = eps

        if self.low_frequency_sigma <= 0.0:
            raise ValueError("low_frequency_sigma must be positive.")
        if self.high_frequency_cutoff < 0.0 or self.high_frequency_cutoff > 1.0:
            raise ValueError("high_frequency_cutoff must be in the range [0.0, 1.0].")
        if self.low_frequency_gain < 0.0:
            raise ValueError("low_frequency_gain must be non-negative.")
        if self.high_frequency_gain < 0.0:
            raise ValueError("high_frequency_gain must be non-negative.")
        if self.dynamic_spectral_gains and self.gain_projector_rank <= 0:
            raise ValueError("gain_projector_rank must be positive.")

        if self.dynamic_spectral_gains:
            r = self.gain_projector_rank
            self.spectral_projector = nn.Sequential(
                nn.Linear(2 * self.state_size, r),
                nn.ReLU(),
                nn.Linear(r, self.state_size),
            )
            nn.init.zeros_(self.spectral_projector[-1].weight)
            nn.init.zeros_(self.spectral_projector[-1].bias)
            # Keep the dynamic branch functionally inert at initialization by
            # zeroing the projector head, while using a nonzero scale so the
            # projector receives gradients on the first training step.
            self.alpha_spectral = nn.Parameter(torch.ones(()))
        else:
            self.spectral_projector = None
            self.register_parameter("alpha_spectral", None)

    def _fixed_spectral_gain(self, normalized_frequency: Tensor) -> Tensor:
        low_boost = self.low_frequency_gain * torch.exp(
            -0.5 * (normalized_frequency / self.low_frequency_sigma).square()
        )
        transition_width = max(0.05, 0.25 * max(0.05, 1.0 - self.high_frequency_cutoff))
        high_damp = self.high_frequency_gain * torch.sigmoid(
            (normalized_frequency - self.high_frequency_cutoff) / transition_width
        )
        return (1.0 + low_boost - high_damp).clamp_min(0.0)

    def _dynamic_spectral_delta(self, signal: Tensor) -> Tensor:
        if self.spectral_projector is None:
            raise ValueError(
                "Dynamic spectral gains are enabled, but spectral_projector is not initialized."
            )

        s_flat = signal.reshape(signal.shape[0], -1)
        if s_flat.shape[-1] != self.state_size:
            raise ValueError(
                "Dynamic spectral gain signal has "
                f"{s_flat.shape[-1]} elements, expected {self.state_size}."
            )
        gain_features = torch.cat([s_flat.real, s_flat.imag], dim=-1)
        delta = self.spectral_projector(gain_features)
        return delta.to(s_flat.real.dtype)

    def _radial_spectral_delta(self, delta: Tensor, normalized_frequency: Tensor) -> Tensor:
        radial_positions = normalized_frequency.to(delta.dtype).reshape(-1).clamp(0.0, 1.0)
        if self.state_size == 1:
            interpolated = delta[:, :1].expand(delta.shape[0], radial_positions.numel())
            return interpolated.reshape(delta.shape[0], *normalized_frequency.shape)

        scaled = radial_positions * (self.state_size - 1)
        lower = scaled.floor().to(torch.long)
        upper = scaled.ceil().to(torch.long)
        fraction = (scaled - lower.to(scaled.dtype)).unsqueeze(0)
        lower_delta = delta.index_select(dim=1, index=lower)
        upper_delta = delta.index_select(dim=1, index=upper)
        interpolated = lower_delta + (upper_delta - lower_delta) * fraction
        return interpolated.reshape(delta.shape[0], *normalized_frequency.shape)

    def _spectral_gain(self, normalized_frequency: Tensor, signal: Optional[Tensor] = None) -> Tensor:
        base_gain = self._fixed_spectral_gain(normalized_frequency)
        if not self.dynamic_spectral_gains or signal is None:
            return base_gain

        delta = self._dynamic_spectral_delta(signal)
        if self.anisotropic_spectral_gains:
            delta = delta.reshape(signal.shape[0], *self.state_shape)
        else:
            delta = self._radial_spectral_delta(delta, normalized_frequency)
        gain = base_gain.unsqueeze(0) + self.alpha_spectral.to(delta.dtype) * delta
        return gain.clamp_min(0.0)

    def _band_delta(
        self,
        signal: Tensor,
        *,
        low: float,
        high: float,
    ) -> Tensor:
        delta = self._dynamic_spectral_delta(signal)
        positions = (
            torch.arange(self.state_size, device=delta.device, dtype=delta.dtype) + 0.5
        ) / self.state_size
        if high >= 1.0:
            band_mask = (positions >= low) & (positions <= high)
        else:
            band_mask = (positions >= low) & (positions < high)
        if not band_mask.any().item():
            center = 0.5 * (low + high)
            nearest = torch.argmin((positions - center).abs()).reshape(1)
            band_mask = torch.zeros_like(positions, dtype=torch.bool)
            band_mask[nearest] = True
        return delta[:, band_mask].mean(dim=-1, keepdim=True)

    def _band_gain(
        self,
        *,
        low: float,
        high: float,
        device: torch.device,
        dtype: torch.dtype,
        signal: Optional[Tensor] = None,
    ) -> Tensor:
        center_frequency = torch.tensor(
            0.5 * (low + high),
            device=device,
            dtype=dtype,
        )
        base_gain = self._fixed_spectral_gain(center_frequency)
        if not self.dynamic_spectral_gains or signal is None:
            return base_gain

        delta = self._band_delta(signal, low=low, high=high)
        return (base_gain + self.alpha_spectral.to(delta.dtype) * delta).clamp_min(0.0)


class FFTSpectralCoupling(_SpectralCouplingBase):
    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.ndim < 2:
            raise ValueError("Expected tensor to have a batch axis and at least one state axis.")

        state_dims = tuple(range(1, tensor.ndim))
        frequency_bands = [torch.fft.fftfreq(size, device=tensor.device) for size in tensor.shape[1:]]
        mesh = torch.meshgrid(*frequency_bands, indexing="ij")
        radius = torch.sqrt(sum(component.square() for component in mesh))
        max_radius = radius.max().clamp_min(self.eps)
        radial_frequency = radius / max_radius

        signal_for_gain = tensor if self.dynamic_spectral_gains else None
        filtered = torch.fft.fftn(tensor, dim=state_dims) * self._spectral_gain(
            radial_frequency,
            signal=signal_for_gain,
        )
        return torch.fft.ifftn(filtered, dim=state_dims).to(tensor.dtype)


def _haar_split(x: Tensor) -> tuple[Tensor, Tensor, int]:
    original_length = x.shape[-1]
    if original_length == 1:
        zeros = torch.zeros_like(x)
        return x, zeros, original_length

    if original_length % 2 == 1:
        x = torch.cat([x, x[..., -1:]], dim=-1)

    pairs = x.reshape(*x.shape[:-1], -1, 2)
    scale = math.sqrt(0.5)
    approx = scale * (pairs[..., 0] + pairs[..., 1])
    detail = scale * (pairs[..., 0] - pairs[..., 1])
    return approx, detail, original_length


def _haar_merge(approx: Tensor, detail: Tensor, original_length: int) -> Tensor:
    scale = math.sqrt(0.5)
    merged = torch.stack(
        [
            scale * (approx + detail),
            scale * (approx - detail),
        ],
        dim=-1,
    ).reshape(*approx.shape[:-1], -1)
    return merged[..., :original_length]


@dataclass
class _WaveletNode:
    coeffs: Tensor
    original_length: int
    band_low: float
    band_high: float
    approx: Optional["_WaveletNode"] = None
    detail: Optional["_WaveletNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.approx is None and self.detail is None


class _WaveletCouplingBase(_SpectralCouplingBase):
    def __init__(
        self,
        state_shape: Sequence[int],
        *,
        low_frequency_gain: float,
        low_frequency_sigma: float,
        high_frequency_gain: float,
        high_frequency_cutoff: float,
        wavelet_levels: Optional[int],
        dynamic_spectral_gains: bool = False,
        anisotropic_spectral_gains: bool = False,
        gain_projector_rank: int = 8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            state_shape,
            low_frequency_gain=low_frequency_gain,
            low_frequency_sigma=low_frequency_sigma,
            high_frequency_gain=high_frequency_gain,
            high_frequency_cutoff=high_frequency_cutoff,
            dynamic_spectral_gains=dynamic_spectral_gains,
            anisotropic_spectral_gains=anisotropic_spectral_gains,
            gain_projector_rank=gain_projector_rank,
            eps=eps,
        )
        if wavelet_levels is not None and wavelet_levels <= 0:
            raise ValueError("wavelet_levels must be positive when provided.")
        self.wavelet_levels = wavelet_levels

    def _flatten(self, tensor: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        return tensor.reshape(tensor.shape[0], -1), tuple(tensor.shape[1:])

    def _reshape(self, vector: Tensor, original_shape: tuple[int, ...]) -> Tensor:
        return vector.reshape(vector.shape[0], *original_shape)

    def _max_levels(self, signal_length: int) -> int:
        levels = 0
        current = signal_length
        while current > 1:
            current = (current + 1) // 2
            levels += 1
        if self.wavelet_levels is None:
            return levels
        return min(levels, self.wavelet_levels)


class DWTSpectralCoupling(_WaveletCouplingBase):
    def forward(self, tensor: Tensor) -> Tensor:
        flat, original_shape = self._flatten(tensor)
        levels = self._max_levels(flat.shape[-1])
        if levels == 0:
            return tensor

        approx = flat
        details: list[tuple[Tensor, int, float, float]] = []
        for level in range(levels):
            approx, detail, original_length = _haar_split(approx)
            band_low = 2.0 ** (-(level + 1))
            band_high = 2.0 ** (-level)
            details.append((detail, original_length, band_low, band_high))

        approx_gain = self._band_gain(
            low=0.0,
            high=2.0**-levels,
            device=flat.device,
            dtype=flat.real.dtype,
            signal=flat if self.dynamic_spectral_gains else None,
        )
        reconstructed = approx * approx_gain.to(approx.dtype)

        for detail, original_length, band_low, band_high in reversed(details):
            gain = self._band_gain(
                low=band_low,
                high=band_high,
                device=detail.device,
                dtype=detail.real.dtype,
                signal=flat if self.dynamic_spectral_gains else None,
            )
            reconstructed = _haar_merge(reconstructed, detail * gain.to(detail.dtype), original_length)

        return self._reshape(reconstructed, original_shape).to(tensor.dtype)


class WaveletPacketSpectralCoupling(_WaveletCouplingBase):
    def __init__(
        self,
        state_shape: Sequence[int],
        *,
        low_frequency_gain: float,
        low_frequency_sigma: float,
        high_frequency_gain: float,
        high_frequency_cutoff: float,
        wavelet_levels: Optional[int],
        phase_aware_best_basis: bool,
        dynamic_spectral_gains: bool = False,
        anisotropic_spectral_gains: bool = False,
        gain_projector_rank: int = 8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            state_shape,
            low_frequency_gain=low_frequency_gain,
            low_frequency_sigma=low_frequency_sigma,
            high_frequency_gain=high_frequency_gain,
            high_frequency_cutoff=high_frequency_cutoff,
            wavelet_levels=wavelet_levels,
            dynamic_spectral_gains=dynamic_spectral_gains,
            anisotropic_spectral_gains=anisotropic_spectral_gains,
            gain_projector_rank=gain_projector_rank,
            eps=eps,
        )
        self.phase_aware_best_basis = phase_aware_best_basis

    def _build_packet_tree(
        self,
        coeffs: Tensor,
        *,
        level: int,
        max_levels: int,
        band_low: float,
        band_high: float,
    ) -> _WaveletNode:
        node = _WaveletNode(
            coeffs=coeffs,
            original_length=coeffs.shape[-1],
            band_low=band_low,
            band_high=band_high,
        )
        if level >= max_levels or coeffs.shape[-1] <= 1:
            return node

        approx, detail, _ = _haar_split(coeffs)
        midpoint = 0.5 * (band_low + band_high)
        node.approx = self._build_packet_tree(
            approx,
            level=level + 1,
            max_levels=max_levels,
            band_low=band_low,
            band_high=midpoint,
        )
        node.detail = self._build_packet_tree(
            detail,
            level=level + 1,
            max_levels=max_levels,
            band_low=midpoint,
            band_high=band_high,
        )
        return node

    def _energy_entropy(self, coeffs: Tensor) -> Tensor:
        energy = coeffs.abs().square().reshape(-1)
        total_energy = energy.sum()
        if total_energy.item() <= self.eps:
            return energy.new_zeros(())

        probabilities = energy / total_energy
        safe_probabilities = probabilities.clamp_min(self.eps)
        return -(probabilities * safe_probabilities.log()).sum()

    def _phase_coherence(self, coeffs: Tensor, global_phase: Tensor) -> Tensor:
        if coeffs.numel() == 0:
            return coeffs.real.new_ones(())

        magnitudes = coeffs.abs()
        normalized = torch.where(
            magnitudes > self.eps,
            coeffs / magnitudes.clamp_min(self.eps),
            torch.ones_like(coeffs),
        )
        aligned = normalized * global_phase.expand_as(coeffs).conj()
        return aligned.mean().abs().real.clamp(0.0, 1.0)

    def _leaf_cost(self, coeffs: Tensor, global_phase: Tensor) -> Tensor:
        entropy = self._energy_entropy(coeffs)
        if not self.phase_aware_best_basis:
            return entropy
        return entropy + (1.0 - self._phase_coherence(coeffs, global_phase))

    def _require_children(self, node: _WaveletNode, operation: str) -> Tuple[_WaveletNode, _WaveletNode]:
        if node.approx is None or node.detail is None:
            raise ValueError(
                "Wavelet packet tree invariant violated during "
                f"{operation}: non-leaf nodes must have both approx and detail children."
            )
        return node.approx, node.detail

    def _select_best_basis(self, node: _WaveletNode, global_phase: Tensor) -> Tensor:
        leaf_cost = self._leaf_cost(node.coeffs, global_phase)
        if node.is_leaf:
            return leaf_cost

        approx, detail = self._require_children(node, "best-basis selection")
        split_cost = self._select_best_basis(approx, global_phase) + self._select_best_basis(
            detail,
            global_phase,
        )
        if split_cost.item() < leaf_cost.item():
            return split_cost

        node.approx = None
        node.detail = None
        return leaf_cost

    def _filter_leaves(self, node: _WaveletNode, signal: Optional[Tensor] = None) -> None:
        if node.is_leaf:
            gain = self._band_gain(
                low=node.band_low,
                high=node.band_high,
                device=node.coeffs.device,
                dtype=node.coeffs.real.dtype,
                signal=signal if self.dynamic_spectral_gains else None,
            )
            node.coeffs = node.coeffs * gain.to(node.coeffs.dtype)
            return

        approx, detail = self._require_children(node, "leaf filtering")
        self._filter_leaves(approx, signal)
        self._filter_leaves(detail, signal)

    def _reconstruct(self, node: _WaveletNode) -> Tensor:
        if node.is_leaf:
            return node.coeffs

        approx_node, detail_node = self._require_children(node, "reconstruction")
        approx = self._reconstruct(approx_node)
        detail = self._reconstruct(detail_node)
        return _haar_merge(approx, detail, node.original_length)

    def forward(self, tensor: Tensor) -> Tensor:
        flat, original_shape = self._flatten(tensor)
        levels = self._max_levels(flat.shape[-1])
        if levels == 0:
            return tensor

        phase_reference = flat.sum(dim=-1, keepdim=True)
        phase_reference = torch.where(
            phase_reference.abs() > self.eps,
            phase_reference / phase_reference.abs().clamp_min(self.eps),
            torch.ones_like(phase_reference),
        )

        tree = self._build_packet_tree(
            flat,
            level=0,
            max_levels=levels,
            band_low=0.0,
            band_high=1.0,
        )
        self._select_best_basis(tree, phase_reference)
        self._filter_leaves(tree, flat)
        reconstructed = self._reconstruct(tree)
        return self._reshape(reconstructed, original_shape).to(tensor.dtype)


class CrossDimensionalBilinear(nn.Module):
    def __init__(self, state_size: int, rank: int = 8) -> None:
        super().__init__()
        self.U = ComplexLinear(state_size, rank, bias=False)
        self.V = ComplexLinear(state_size, rank, bias=False)
        self.W = ComplexLinear(rank, state_size, bias=False)
        # Zero only the output head: the residual is inert at init, while U/V
        # remain live once W receives its first gradient update.
        nn.init.zeros_(self.W.weight_real)
        nn.init.zeros_(self.W.weight_imag)

    def forward(self, z_flat: Tensor) -> Tensor:
        p = self.U(z_flat)
        q = self.V(z_flat)
        gamma = p * q.conj()
        return self.W(gamma)


class ReciprocatorMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        state_shape: Sequence[int],
        enable_self_relation: bool = False,
        enable_dynamic_gains: bool = False,
        gain_projector_rank: int = 8,
        coupling_type: str = "sequential",
        low_frequency_gain: float = 0.5,
        low_frequency_sigma: float = 0.35,
        high_frequency_gain: float = 0.5,
        high_frequency_cutoff: float = 0.5,
        dynamic_spectral_gains: bool = False,
        anisotropic_spectral_gains: bool = False,
        wavelet_levels: Optional[int] = None,
        phase_scale: float = math.pi,
        normalization_type: str = "frobenius",
        enable_cross_bilinear: bool = True,
        cross_bilinear_rank: int = 8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self.state_size = math.prod(self.state_shape)
        self.enable_self_relation = enable_self_relation
        self.enable_dynamic_gains = enable_dynamic_gains
        self.gain_projector_rank = gain_projector_rank
        self.coupling_type = canonicalize_coupling_type(coupling_type)
        self.dynamic_spectral_gains = dynamic_spectral_gains
        self.anisotropic_spectral_gains = anisotropic_spectral_gains
        self.normalization_type = canonicalize_normalization_type(normalization_type)
        self.enable_cross_bilinear = enable_cross_bilinear
        self.cross_bilinear_rank = cross_bilinear_rank
        self.eps = eps

        self.pre_norm = ComplexLayerNorm(hidden_size, eps=eps)
        self.signal_projector = TensorSignalProjector(
            hidden_size=hidden_size,
            state_shape=self.state_shape,
            phase_scale=phase_scale,
            normalization_type=self.normalization_type,
            eps=eps,
        )
        if self.coupling_type == "sequential":
            self.coupling = SequentialModeCoupling(self.state_shape, eps=eps)
        elif self.coupling_type == "fft":
            self.coupling = FFTSpectralCoupling(
                self.state_shape,
                low_frequency_gain=low_frequency_gain,
                low_frequency_sigma=low_frequency_sigma,
                high_frequency_gain=high_frequency_gain,
                high_frequency_cutoff=high_frequency_cutoff,
                dynamic_spectral_gains=dynamic_spectral_gains,
                anisotropic_spectral_gains=anisotropic_spectral_gains,
                gain_projector_rank=gain_projector_rank,
                eps=eps,
            )
        elif self.coupling_type == "dwt":
            self.coupling = DWTSpectralCoupling(
                self.state_shape,
                low_frequency_gain=low_frequency_gain,
                low_frequency_sigma=low_frequency_sigma,
                high_frequency_gain=high_frequency_gain,
                high_frequency_cutoff=high_frequency_cutoff,
                wavelet_levels=wavelet_levels,
                dynamic_spectral_gains=dynamic_spectral_gains,
                anisotropic_spectral_gains=anisotropic_spectral_gains,
                gain_projector_rank=gain_projector_rank,
                eps=eps,
            )
        else:
            self.coupling = WaveletPacketSpectralCoupling(
                self.state_shape,
                low_frequency_gain=low_frequency_gain,
                low_frequency_sigma=low_frequency_sigma,
                high_frequency_gain=high_frequency_gain,
                high_frequency_cutoff=high_frequency_cutoff,
                wavelet_levels=wavelet_levels,
                phase_aware_best_basis=self.coupling_type == "wavelet_packet_max_gauge",
                dynamic_spectral_gains=dynamic_spectral_gains,
                anisotropic_spectral_gains=anisotropic_spectral_gains,
                gain_projector_rank=gain_projector_rank,
                eps=eps,
            )

        self.decay_logit = nn.Parameter(torch.zeros(self.state_shape))
        self.input_logit = nn.Parameter(torch.zeros(self.state_shape))
        self.recurrent_logit = nn.Parameter(torch.zeros(self.state_shape))
        if self.enable_self_relation:
            self.self_relation_logit = nn.Parameter(torch.zeros(self.state_shape))
        else:
            self.register_parameter("self_relation_logit", None)

        self.return_map = RealToComplexLinear(3 * self.state_size, hidden_size)
        self.gate = nn.Linear(4 * hidden_size, hidden_size)
        if self.enable_dynamic_gains:
            r = self.gain_projector_rank
            self.gain_projector = nn.Sequential(
                nn.Linear(2 * self.state_size, r),
                nn.ReLU(),
                nn.Linear(r, 3 * self.state_size),
            )
            nn.init.zeros_(self.gain_projector[-1].weight)
            nn.init.zeros_(self.gain_projector[-1].bias)
            self.alpha_D = nn.Parameter(torch.zeros(()))
            self.alpha_A = nn.Parameter(torch.zeros(()))
            self.alpha_B = nn.Parameter(torch.zeros(()))
        else:
            self.gain_projector = None
            self.register_parameter("alpha_D", None)
            self.register_parameter("alpha_A", None)
            self.register_parameter("alpha_B", None)

        # Learned initial state prior — relational seed for t=0 coupling.
        # Small complex random init, immediately normalized over the state axes.
        _init_raw = torch.randn(self.state_shape, dtype=torch.cfloat) * 0.01
        _init_batched = _init_raw.unsqueeze(0)
        self.initial_state_param = nn.Parameter(
            normalize_complex(
                _init_batched,
                normalization_type=self.normalization_type,
                dims=tuple(range(1, _init_batched.ndim)),
                eps=self.eps,
            ).squeeze(0)
        )
        if self.enable_cross_bilinear:
            self.cross_bilinear = CrossDimensionalBilinear(
                self.state_size,
                rank=self.cross_bilinear_rank,
            )
        else:
            self.cross_bilinear = None

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return the learned initial state prior, broadcast over the batch."""
        state = self.initial_state_param.to(device=device, dtype=dtype)
        if batch_size > 1:
            state = state.unsqueeze(0).expand(batch_size, *state.shape)
        else:
            state = state.unsqueeze(0)
        return state.clone()

    def _state_features(self, state: Tensor) -> Tensor:
        features = phase_aware_feature_map(state, batch_dim=True)
        return features.reshape(state.shape[0], -1)

    def _gain_logits(self, signal: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        decay_logit = self.decay_logit
        input_logit = self.input_logit
        recurrent_logit = self.recurrent_logit

        if self.enable_dynamic_gains:
            s_flat = signal.reshape(signal.shape[0], -1)
            gain_features = torch.cat([s_flat.real, s_flat.imag], dim=-1)

            if self.gain_projector is None:
                raise ValueError(
                    "Dynamic gains are enabled, but gain_projector is not initialized."
                )
            delta = self.gain_projector(gain_features)
            delta = delta.reshape(-1, 3, *self.state_shape)
            delta_D, delta_A, delta_B = delta.unbind(dim=1)

            decay_logit = self.decay_logit + self.alpha_D * delta_D
            input_logit = self.input_logit + self.alpha_A * delta_A
            recurrent_logit = self.recurrent_logit + self.alpha_B * torch.tanh(delta_B)

        return decay_logit, input_logit, recurrent_logit

    def step(self, hidden_t: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if hidden_t.ndim != 2:
            raise ValueError("Expected hidden_t to have shape [batch, hidden_size].")
        if not torch.is_complex(hidden_t):
            raise TypeError("ReciprocatorMixer expects complex-valued hidden states.")

        if state is None:
            state = self.initial_state(hidden_t.shape[0], device=hidden_t.device, dtype=hidden_t.dtype)

        normalized_hidden = self.pre_norm(hidden_t)
        signal = self.signal_projector(normalized_hidden)
        decay_logit, input_logit, recurrent_logit = self._gain_logits(signal)
        relational = signal * state
        routed = self.coupling(relational)
        if self.cross_bilinear is not None:
            z_flat = relational.reshape(relational.shape[0], -1)
            routed = routed + self.cross_bilinear(z_flat).reshape(relational.shape)
        state_dims = tuple(range(1, state.ndim))

        proposal = (
            torch.sigmoid(decay_logit) * state
            + torch.sigmoid(input_logit) * signal
            + torch.tanh(recurrent_logit) * routed
        )
        if self.enable_self_relation:
            tentative_state = normalize_complex(
                proposal,
                normalization_type=self.normalization_type,
                dims=state_dims,
                eps=self.eps,
            )
            self_relation = state * tentative_state
            proposal = proposal + torch.tanh(self.self_relation_logit) * self_relation

        next_state = normalize_complex(
            proposal,
            normalization_type=self.normalization_type,
            dims=state_dims,
            eps=self.eps,
        )

        delta = self.return_map(self._state_features(next_state - state))
        gate_input = torch.cat(
            [normalized_hidden.real, normalized_hidden.imag, delta.real, delta.imag], dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input))
        return gate.to(delta.dtype) * delta, next_state

    def _compute_delta(
        self,
        next_state: Tensor,
        old_state: Tensor,
        normalized_hidden: Tensor,
    ) -> Tensor:
        features = self._state_features(next_state - old_state)
        delta = self.return_map(features)
        gate_input = torch.cat(
            [normalized_hidden.real, normalized_hidden.imag, delta.real, delta.imag], dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input))
        return gate.to(delta.dtype) * delta

    def _chunked_forward(
        self,
        hidden: Tensor,
        state: Optional[Tensor] = None,
        *,
        chunk_size: int = 16,
        track_drift: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[dict]]:
        """Chunked causal forward: exact relational handoffs at boundaries,
        geometric-series core + stale coupling inside chunks.

        K=1 recovers exact sequential. Larger K amortizes coupling cost across tokens.
        Self-relation is boundary-only inside chunks to preserve affine-core linearity.
        """
        B, T, _ = hidden.shape
        if state is None:
            state = self.initial_state(B, device=hidden.device, dtype=hidden.dtype)

        if chunk_size < 1 or chunk_size >= T:
            delta, next_state = self._sequential_forward(hidden, state)
            return delta, next_state, None

        deltas: list[Tensor] = []
        current_state = state
        drifts: Optional[list[float]] = [] if track_drift else None

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = hidden[:, start:end]  # [B, L, D]

            # Exact boundary handoff: full relational update on first token of chunk
            delta_t, current_state = self.step(chunk[:, 0], current_state)
            deltas.append(delta_t)
            chunk_start_state = current_state  # normalized, on Frobenius sphere

            L_rest = end - start - 1
            if L_rest > 0:
                rest = chunk[:, 1:]  # [B, L_rest, D]

                # pre_norm handles arbitrary batch dims (operates on last dim only)
                normed = self.pre_norm(rest)  # [B, L_rest, D]

                # signal_projector assumes [B, D] input (reshapes to state_shape via shape[0])
                # so we must flatten the L dimension, project, then restore
                state_shape = self.state_shape
                normed_flat = normed.reshape(B * L_rest, normed.shape[-1])
                s_flat = self.signal_projector(normed_flat)  # [B*L_rest, *state_shape]
                s_rest = s_flat.reshape(B, L_rest, *state_shape)  # [B, L_rest, *state_shape]

                # Stale relational product: expand chunk_start_state to match flat batch
                stale_flat = (
                    chunk_start_state
                    .unsqueeze(1)
                    .expand(B, L_rest, *state_shape)
                    .reshape(B * L_rest, *state_shape)
                )
                Z_flat = s_flat * stale_flat
                routed_flat = self.coupling(Z_flat)  # [B*L_rest, *state_shape]
                if self.cross_bilinear is not None:
                    z_flat = Z_flat.reshape(Z_flat.shape[0], -1)
                    routed_flat = routed_flat + self.cross_bilinear(z_flat).reshape(
                        Z_flat.shape
                    )
                routed_stale = routed_flat.reshape(B, L_rest, *state_shape)

                if self.enable_dynamic_gains:
                    decay_logit, input_logit, recurrent_logit = self._gain_logits(s_flat)
                    decay_gain = torch.sigmoid(decay_logit).reshape(B, L_rest, *state_shape)
                    input_gain = torch.sigmoid(input_logit).reshape(B, L_rest, *state_shape)
                    recurrent_gain = torch.tanh(recurrent_logit).reshape(B, L_rest, *state_shape)
                else:
                    decay_gain = torch.sigmoid(self.decay_logit).reshape(1, 1, *state_shape).expand(B, L_rest, *state_shape)
                    input_gain = torch.sigmoid(self.input_logit).reshape(1, 1, *state_shape).expand(B, L_rest, *state_shape)
                    recurrent_gain = torch.tanh(self.recurrent_logit).reshape(1, 1, *state_shape).expand(B, L_rest, *state_shape)

                core_state = chunk_start_state
                if track_drift:
                    for i in range(L_rest):
                        old_core_state = core_state
                        proposal = (
                            decay_gain[:, i] * core_state
                            + input_gain[:, i] * s_rest[:, i]
                            + recurrent_gain[:, i] * routed_stale[:, i]
                        )
                        # Self-relation is boundary-only: zeroed inside chunk to keep
                        # the affine core a pure linear recurrence (scannable)
                        core_state = normalize_complex(
                            proposal,
                            normalization_type=self.normalization_type,
                            dims=tuple(range(1, proposal.ndim)),
                            eps=self.eps,
                        )
                        state_delta = core_state - chunk_start_state
                        drifts.append(
                            state_delta.flatten(start_dim=1).norm(p=2, dim=1).mean().item()
                        )
                        delta_t = self._compute_delta(core_state, old_core_state, normed[:, i])
                        deltas.append(delta_t)
                else:
                    for i in range(L_rest):
                        old_core_state = core_state
                        proposal = (
                            decay_gain[:, i] * core_state
                            + input_gain[:, i] * s_rest[:, i]
                            + recurrent_gain[:, i] * routed_stale[:, i]
                        )
                        # Self-relation is boundary-only: zeroed inside chunk to keep
                        # the affine core a pure linear recurrence (scannable)
                        core_state = normalize_complex(
                            proposal,
                            normalization_type=self.normalization_type,
                            dims=tuple(range(1, proposal.ndim)),
                            eps=self.eps,
                        )
                        delta_t = self._compute_delta(core_state, old_core_state, normed[:, i])
                        deltas.append(delta_t)

                current_state = core_state

        drift_stats = None
        if track_drift and drifts:
            drift_stats = {
                "mean_drift": float(sum(drifts) / len(drifts)),
                "max_drift": float(max(drifts)),
                "K": chunk_size,
            }

        return torch.stack(deltas, dim=1), current_state, drift_stats

    def _sequential_forward(self, hidden: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        deltas = []
        for idx in range(hidden.shape[1]):
            delta_t, state = self.step(hidden[:, idx], state)
            deltas.append(delta_t)
        return torch.stack(deltas, dim=1), state

    def forward(
        self,
        hidden: Tensor,
        state: Optional[Tensor] = None,
        *,
        chunk_size: Optional[int] = None,
        track_drift: bool = False,
    ):
        if hidden.ndim != 3:
            raise ValueError("Expected hidden to have shape [batch, seq, hidden_size].")
        if not torch.is_complex(hidden):
            raise TypeError("ReciprocatorMixer expects complex-valued hidden states.")

        if state is None:
            state = self.initial_state(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)

        if chunk_size is None:
            delta, next_state = self._sequential_forward(hidden, state)
            if track_drift:
                return delta, next_state, None
            return delta, next_state

        delta, next_state, drift_stats = self._chunked_forward(
            hidden,
            state,
            chunk_size=chunk_size,
            track_drift=track_drift,
        )
        if track_drift:
            return delta, next_state, drift_stats
        return delta, next_state
