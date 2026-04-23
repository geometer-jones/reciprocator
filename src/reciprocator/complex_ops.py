from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _canonical_dims(ndim: int, dims: Optional[Iterable[int]]) -> Tuple[int, ...]:
    if dims is None:
        return tuple(range(ndim))
    canonical = []
    for dim in dims:
        canonical.append(dim if dim >= 0 else ndim + dim)
    return tuple(canonical)


def _complex_power(x: Tensor) -> Tensor:
    return x.real.square() + x.imag.square()


def canonicalize_normalization_type(normalization_type: str) -> str:
    canonical = normalization_type.strip().lower()
    if canonical not in {"frobenius", "per_mode"}:
        raise ValueError("normalization_type must be one of {'frobenius', 'per_mode'}.")
    return canonical


def frobenius_normalize(x: Tensor, dims: Optional[Iterable[int]] = None, eps: float = 1e-8) -> Tensor:
    canonical_dims = _canonical_dims(x.ndim, dims)
    power = _complex_power(x)
    denom = torch.sqrt(power.sum(dim=canonical_dims, keepdim=True).clamp_min(eps))
    return x / denom


def per_mode_normalize(
    x: Tensor,
    dims: Optional[Iterable[int]] = None,
    eps: float = 1e-8,
    num_iters: int = 8,
) -> Tensor:
    canonical_dims = _canonical_dims(x.ndim, dims)
    if num_iters <= 0:
        raise ValueError("num_iters must be positive.")

    normalized = x
    for _ in range(num_iters):
        for dim in canonical_dims:
            power = _complex_power(normalized)
            denom = torch.sqrt(power.sum(dim=dim, keepdim=True).clamp_min(eps))
            normalized = normalized / denom
    return normalized


def normalize_complex(
    x: Tensor,
    *,
    normalization_type: str = "frobenius",
    dims: Optional[Iterable[int]] = None,
    eps: float = 1e-8,
) -> Tensor:
    canonical = canonicalize_normalization_type(normalization_type)
    if canonical == "frobenius":
        return frobenius_normalize(x, dims=dims, eps=eps)
    return per_mode_normalize(x, dims=dims, eps=eps)


class ComplexLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias_real = nn.Parameter(torch.zeros(hidden_size))
        self.bias_imag = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexLayerNorm expects a complex-valued tensor.")

        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        magnitude_var = centered.real.square() + centered.imag.square()
        scale = torch.rsqrt(magnitude_var.mean(dim=-1, keepdim=True).clamp_min(self.eps))
        normalized = centered * scale
        bias = torch.complex(self.bias_real, self.bias_imag)
        return normalized * self.weight + bias


class RealToComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, 2 * out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        projected = self.proj(x)
        real, imag = projected.chunk(2, dim=-1)
        return torch.complex(real, imag)


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_real, a=5**0.5)
        nn.init.kaiming_uniform_(self.weight_imag, a=5**0.5)
        if self.bias_real is not None and self.bias_imag is not None:
            bound = self.in_features ** -0.5
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexLinear expects a complex-valued tensor.")

        real = F.linear(x.real, self.weight_real, self.bias_real) - F.linear(x.imag, self.weight_imag)
        imag = F.linear(x.real, self.weight_imag, self.bias_imag) + F.linear(x.imag, self.weight_real)
        return torch.complex(real, imag)


class ComplexModReLU(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexModReLU expects a complex-valued tensor.")

        magnitude = x.abs()
        activated = F.relu(magnitude + self.bias)
        scale = activated / magnitude.clamp_min(self.eps)
        return x * scale
