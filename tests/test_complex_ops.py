import torch

from reciprocator.complex_ops import ComplexLayerNorm, frobenius_normalize, per_mode_normalize


def random_complex(*shape: int) -> torch.Tensor:
    return torch.complex(torch.randn(*shape), torch.randn(*shape))


def test_frobenius_normalize_returns_unit_norm_per_batch() -> None:
    x = random_complex(3, 2, 4)
    y = frobenius_normalize(x, dims=(1, 2))
    norms = torch.sqrt((y.real.square() + y.imag.square()).sum(dim=(1, 2)))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_per_mode_normalize_returns_unit_norm_fibers() -> None:
    torch.manual_seed(1)
    x = random_complex(3, 2, 4)
    y = per_mode_normalize(x, dims=(1, 2))
    power = y.real.square() + y.imag.square()

    mode_1_norms = torch.sqrt(power.sum(dim=1))
    mode_2_norms = torch.sqrt(power.sum(dim=2))
    expected_mode_1 = torch.full_like(mode_1_norms, (2.0 / 4.0) ** 0.5)
    expected_mode_2 = torch.ones_like(mode_2_norms)

    assert torch.allclose(mode_1_norms, expected_mode_1, atol=1e-3)
    assert torch.allclose(mode_2_norms, expected_mode_2, atol=1e-5)


def test_complex_layer_norm_centers_and_scales_last_dimension() -> None:
    layer_norm = ComplexLayerNorm(hidden_size=8)
    x = random_complex(4, 8)
    y = layer_norm(x)

    means = y.mean(dim=-1)
    rms = torch.sqrt((y.real.square() + y.imag.square()).mean(dim=-1))

    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
