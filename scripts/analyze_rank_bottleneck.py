#!/usr/bin/env python3
"""Diagnose whether the Hadamard relational step is a rank bottleneck.

The script loads a training checkpoint, replays the model forward pass, and at
each Reciprocator layer/timestep compares the singular spectrum of:

    relational = signal * state
    routed = coupling(relational)

For tensor states, spectra are computed on mode unfoldings:
mode i is reshaped as [state_shape[i], product(other state dimensions)].
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reciprocator.complex_ops import normalize_complex  # noqa: E402
from reciprocator.model import LocalAttentionBlock, ReciprocatorBlock, ReciprocatorLM  # noqa: E402
from reciprocator.training import (  # noqa: E402
    CharTokenizer,
    TextDataset,
    TrainingConfig,
    build_corpus_dataset,
    sample_causal_lm_batch,
)


@dataclass(frozen=True)
class SpectrumMetrics:
    effective_rank: float
    stable_rank: float
    top_singular_mass: float
    singular_values: list[float]


@dataclass(frozen=True)
class RankObservation:
    layer: int
    timestep: int
    mode: int
    matrix_shape: Tuple[int, int]
    max_rank: int
    hadamard: SpectrumMetrics
    coupled: SpectrumMetrics
    recovered_rank: float
    recovery_ratio: Optional[float]
    recovery_fraction_of_possible: float
    hadamard_near_rank1: bool


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: ReciprocatorLM
    tokenizer: CharTokenizer
    config: Mapping[str, Any]
    dataset: Optional[TextDataset]
    warnings: list[str]


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
    return CharTokenizer(
        stoi={character: index for index, character in enumerate(vocabulary)},
        itos=tuple(vocabulary),
    )


def _default_config() -> TrainingConfig:
    return TrainingConfig()


def _config_value(config: Mapping[str, Any], name: str) -> Any:
    return config.get(name, getattr(_default_config(), name))


def _resolve_dataset_and_frequencies(
    payload: Mapping[str, Any],
    tokenizer: CharTokenizer,
) -> Tuple[Optional[TextDataset], Tensor, list[str]]:
    warnings: list[str] = []
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return None, torch.ones(tokenizer.vocab_size), ["Checkpoint is missing config; using uniform token frequencies."]

    corpus_name = config.get("corpus_name") or payload.get("source_name")
    if not isinstance(corpus_name, str):
        return None, torch.ones(tokenizer.vocab_size), ["Checkpoint has no corpus name; using uniform token frequencies."]

    try:
        dataset = build_corpus_dataset(
            corpus_name,
            max_chars=config.get("max_chars"),
            val_fraction=float(config.get("val_fraction", _default_config().val_fraction)),
            tokenizer=tokenizer,
        )
    except Exception as exc:  # pragma: no cover - exact filesystem failure is environment-specific.
        warnings.append(f"Could not rebuild corpus dataset ({exc}); using uniform token frequencies.")
        return None, torch.ones(tokenizer.vocab_size), warnings

    return dataset, dataset.token_frequencies(), warnings


def model_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    tokenizer: CharTokenizer,
    token_frequencies: Tensor,
    device: torch.device,
) -> ReciprocatorLM:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint is missing a config dict.")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")

    model = ReciprocatorLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(_config_value(config, "hidden_size")),
        state_shape=tuple(_config_value(config, "state_shape")),
        num_layers=int(_config_value(config, "num_layers")),
        ffn_expansion_factor=int(_config_value(config, "ffn_expansion_factor")),
        readout_type=str(_config_value(config, "readout_type")),
        token_magnitude_type=str(_config_value(config, "token_magnitude_type")),
        phase_type=str(_config_value(config, "phase_type")),
        token_phase=str(_config_value(config, "token_phase")),
        enable_self_relation=bool(_config_value(config, "enable_self_relation")),
        dynamic_gains=bool(_config_value(config, "dynamic_gains")),
        gain_projector_rank=int(_config_value(config, "gain_projector_rank")),
        enable_cross_layer_state=bool(_config_value(config, "enable_cross_layer_state")),
        coupling_type=str(_config_value(config, "coupling_type")),
        low_frequency_gain=float(_config_value(config, "low_frequency_gain")),
        low_frequency_sigma=float(_config_value(config, "low_frequency_sigma")),
        high_frequency_gain=float(_config_value(config, "high_frequency_gain")),
        high_frequency_cutoff=float(_config_value(config, "high_frequency_cutoff")),
        dynamic_spectral_gains=bool(_config_value(config, "dynamic_spectral_gains")),
        anisotropic_spectral_gains=bool(_config_value(config, "anisotropic_spectral_gains")),
        wavelet_levels=_config_value(config, "wavelet_levels"),
        normalization_type=str(_config_value(config, "normalization_type")),
        token_frequencies=token_frequencies,
        attention_every_k=int(_config_value(config, "attention_every_k")),
        attention_num_heads=int(_config_value(config, "attention_num_heads")),
        attention_window=int(_config_value(config, "attention_window")),
        attention_position=str(_config_value(config, "attention_position")),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_trained_model(checkpoint: Path, *, device: torch.device) -> LoadedCheckpoint:
    payload = load_checkpoint(checkpoint)
    tokenizer = tokenizer_from_checkpoint(payload)
    dataset, token_frequencies, warnings = _resolve_dataset_and_frequencies(payload, tokenizer)
    model = model_from_checkpoint(
        payload,
        tokenizer=tokenizer,
        token_frequencies=token_frequencies,
        device=device,
    )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        config = {}
    return LoadedCheckpoint(model=model, tokenizer=tokenizer, config=config, dataset=dataset, warnings=warnings)


def _mode_unfoldings(tensor: Tensor, modes: Optional[Sequence[int]] = None) -> Iterable[Tuple[int, Tensor]]:
    if tensor.ndim < 2:
        raise ValueError("Expected tensor with shape [batch, *state_shape].")
    rank = tensor.ndim - 1
    selected_modes = tuple(range(rank)) if modes is None else tuple(modes)
    for mode in selected_modes:
        if mode < 0 or mode >= rank:
            raise ValueError(f"Mode {mode} is outside tensor rank {rank}.")
        rows = tensor.shape[mode + 1]
        unfolded = tensor.movedim(mode + 1, 1).reshape(tensor.shape[0], rows, -1)
        yield mode, unfolded


def spectrum_metrics(matrix: Tensor, *, eps: float = 1e-12) -> SpectrumMetrics:
    if matrix.ndim != 2:
        raise ValueError("Expected a single matrix with shape [rows, cols].")
    singular_values = torch.linalg.svdvals(matrix.to("cpu"))
    singular_values = singular_values.real.to(torch.float64)
    total = singular_values.sum()
    if float(total) <= eps:
        return SpectrumMetrics(
            effective_rank=0.0,
            stable_rank=0.0,
            top_singular_mass=0.0,
            singular_values=[float(value) for value in singular_values.tolist()],
        )

    probabilities = singular_values / total.clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    effective_rank = float(torch.exp(entropy).item())

    largest = singular_values[0].clamp_min(eps)
    stable_rank = float((singular_values.square().sum() / largest.square()).item())
    top_singular_mass = float((singular_values[0] / total.clamp_min(eps)).item())
    return SpectrumMetrics(
        effective_rank=effective_rank,
        stable_rank=stable_rank,
        top_singular_mass=top_singular_mass,
        singular_values=[float(value) for value in singular_values.tolist()],
    )


def _mean_spectrum_metrics(matrices: Tensor, *, eps: float = 1e-12) -> SpectrumMetrics:
    per_batch = [spectrum_metrics(matrices[index], eps=eps) for index in range(matrices.shape[0])]
    if not per_batch:
        raise ValueError("Cannot summarize an empty matrix batch.")

    spectrum_len = len(per_batch[0].singular_values)
    mean_spectrum = [
        sum(metrics.singular_values[index] for metrics in per_batch) / len(per_batch)
        for index in range(spectrum_len)
    ]
    return SpectrumMetrics(
        effective_rank=sum(metrics.effective_rank for metrics in per_batch) / len(per_batch),
        stable_rank=sum(metrics.stable_rank for metrics in per_batch) / len(per_batch),
        top_singular_mass=sum(metrics.top_singular_mass for metrics in per_batch) / len(per_batch),
        singular_values=mean_spectrum,
    )


def compare_rank(
    relational: Tensor,
    coupled: Tensor,
    *,
    layer: int,
    timestep: int,
    modes: Optional[Sequence[int]],
    rank1_erank_threshold: float,
    rank1_top_mass_threshold: float,
    eps: float = 1e-12,
) -> list[RankObservation]:
    observations: list[RankObservation] = []
    coupled_by_mode = dict(_mode_unfoldings(coupled, modes))
    for mode, relational_unfolded in _mode_unfoldings(relational, modes):
        coupled_unfolded = coupled_by_mode[mode]
        hadamard_metrics = _mean_spectrum_metrics(relational_unfolded, eps=eps)
        coupled_metrics = _mean_spectrum_metrics(coupled_unfolded, eps=eps)
        max_rank = min(relational_unfolded.shape[-2], relational_unfolded.shape[-1])
        recovered_rank = coupled_metrics.effective_rank - hadamard_metrics.effective_rank
        recovery_ratio = None
        if hadamard_metrics.effective_rank > eps:
            recovery_ratio = coupled_metrics.effective_rank / hadamard_metrics.effective_rank
        recoverable = max_rank - hadamard_metrics.effective_rank
        positive_recovery = max(recovered_rank, 0.0)
        recovery_fraction = positive_recovery / recoverable if recoverable > eps else 0.0
        near_rank1 = (
            hadamard_metrics.effective_rank <= rank1_erank_threshold
            or hadamard_metrics.top_singular_mass >= rank1_top_mass_threshold
        )
        observations.append(
            RankObservation(
                layer=layer,
                timestep=timestep,
                mode=mode,
                matrix_shape=(int(relational_unfolded.shape[-2]), int(relational_unfolded.shape[-1])),
                max_rank=int(max_rank),
                hadamard=hadamard_metrics,
                coupled=coupled_metrics,
                recovered_rank=float(recovered_rank),
                recovery_ratio=float(recovery_ratio),
                recovery_fraction_of_possible=float(recovery_fraction),
                hadamard_near_rank1=bool(near_rank1),
            )
        )
    return observations


def _trace_reciprocator_block(
    block: ReciprocatorBlock,
    hidden: Tensor,
    state: Optional[Tensor],
    *,
    layer: int,
    modes: Optional[Sequence[int]],
    rank1_erank_threshold: float,
    rank1_top_mass_threshold: float,
) -> Tuple[Tensor, Tensor, list[RankObservation]]:
    mixer = block.mixer
    if state is None:
        state = mixer.initial_state(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)

    deltas: list[Tensor] = []
    observations: list[RankObservation] = []
    state_dims = tuple(range(1, state.ndim))

    for timestep in range(hidden.shape[1]):
        hidden_t = hidden[:, timestep]
        normalized_hidden = mixer.pre_norm(hidden_t)
        signal = mixer.signal_projector(normalized_hidden)
        decay_logit, input_logit, recurrent_logit = mixer._gain_logits(signal)
        relational = signal * state
        routed = mixer.coupling(relational)

        observations.extend(
            compare_rank(
                relational.detach(),
                routed.detach(),
                layer=layer,
                timestep=timestep,
                modes=modes,
                rank1_erank_threshold=rank1_erank_threshold,
                rank1_top_mass_threshold=rank1_top_mass_threshold,
            )
        )

        proposal = (
            torch.sigmoid(decay_logit) * state
            + torch.sigmoid(input_logit) * signal
            + torch.tanh(recurrent_logit) * routed
        )
        if mixer.enable_self_relation:
            tentative_state = normalize_complex(
                proposal,
                normalization_type=mixer.normalization_type,
                dims=state_dims,
                eps=mixer.eps,
            )
            self_relation = state * tentative_state
            proposal = proposal + torch.tanh(mixer.self_relation_logit) * self_relation

        next_state = normalize_complex(
            proposal,
            normalization_type=mixer.normalization_type,
            dims=state_dims,
            eps=mixer.eps,
        )
        delta_t = mixer._compute_delta(next_state, state, normalized_hidden)
        deltas.append(delta_t)
        state = next_state

    delta = torch.stack(deltas, dim=1)
    hidden = hidden + block.ffn(block.ffn_norm(delta))
    return hidden, state, observations


def trace_rank_observations(
    model: ReciprocatorLM,
    token_ids: Tensor,
    *,
    modes: Optional[Sequence[int]],
    rank1_erank_threshold: float,
    rank1_top_mass_threshold: float,
) -> Tuple[Tensor, tuple[Optional[Tensor], ...], list[RankObservation]]:
    model.eval()
    next_states: list[Optional[Tensor]] = []
    observations: list[RankObservation] = []
    donor_layer_index = 0

    with torch.no_grad():
        hidden = model.token_lift(token_ids)
        for block in model.blocks:
            if isinstance(block, ReciprocatorBlock):
                hidden, next_state, block_observations = _trace_reciprocator_block(
                    block,
                    hidden,
                    None,
                    layer=donor_layer_index,
                    modes=modes,
                    rank1_erank_threshold=rank1_erank_threshold,
                    rank1_top_mass_threshold=rank1_top_mass_threshold,
                )
                observations.extend(block_observations)
                next_states.append(next_state)
                if model.enable_cross_layer_state:
                    hidden = model._inject_cross_layer_state(
                        hidden,
                        next_state,
                        donor_layer_index=donor_layer_index,
                    )
                donor_layer_index += 1
                continue

            if not isinstance(block, LocalAttentionBlock):
                raise TypeError(f"Unsupported block type in model.blocks: {type(block)!r}")
            hidden, next_state = block(hidden, None)
            next_states.append(next_state)

        logits = model.readout(model.final_norm(hidden))

    return logits, tuple(next_states), observations


def _read_probe_text(args: argparse.Namespace) -> Optional[str]:
    if args.text is not None and args.text_file is not None:
        raise ValueError("Use only one of --text or --text-file.")
    if args.text_file is not None:
        return Path(args.text_file).expanduser().read_text(encoding="utf-8")
    return args.text


def build_probe_batch(
    loaded: LoadedCheckpoint,
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> Tensor:
    text = _read_probe_text(args)
    if text is not None:
        if args.seq_len is not None:
            text = text[: args.seq_len]
        tokens = loaded.tokenizer.encode(text).unsqueeze(0)
        return tokens.to(device)

    if loaded.dataset is None:
        raise ValueError("No probe text was provided and the checkpoint corpus could not be rebuilt.")

    split_tokens = loaded.dataset.val_tokens if args.split == "val" else loaded.dataset.train_tokens
    if split_tokens.numel() <= 1:
        raise ValueError(f"{args.split} split is too small to build a probe batch.")

    config_seq_len = int(_config_value(loaded.config, "seq_len"))
    seq_len = args.seq_len if args.seq_len is not None else min(config_seq_len, 128)
    seq_len = min(seq_len, int(split_tokens.numel()) - 1)
    batch_size = args.batch_size if args.batch_size is not None else min(int(_config_value(loaded.config, "batch_size")), 4)
    torch.manual_seed(args.seed)
    inputs, _ = sample_causal_lm_batch(split_tokens, batch_size, seq_len, device=device)
    return inputs


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def summarize_observations(observations: Sequence[RankObservation]) -> dict[str, Any]:
    if not observations:
        raise ValueError("No rank observations were collected.")

    hadamard_ranks = [obs.hadamard.effective_rank for obs in observations]
    coupled_ranks = [obs.coupled.effective_rank for obs in observations]
    recovered = [obs.recovered_rank for obs in observations]
    recovery_fractions = [obs.recovery_fraction_of_possible for obs in observations]
    bottleneck_fraction = sum(obs.hadamard_near_rank1 for obs in observations) / len(observations)

    by_layer_mode = []
    keys = sorted({(obs.layer, obs.mode) for obs in observations})
    for layer, mode in keys:
        group = [obs for obs in observations if obs.layer == layer and obs.mode == mode]
        by_layer_mode.append(
            {
                "layer": layer,
                "mode": mode,
                "matrix_shape": group[0].matrix_shape,
                "max_rank": group[0].max_rank,
                "hadamard_effective_rank_mean": _mean([obs.hadamard.effective_rank for obs in group]),
                "coupled_effective_rank_mean": _mean([obs.coupled.effective_rank for obs in group]),
                "recovered_rank_mean": _mean([obs.recovered_rank for obs in group]),
                "recovery_fraction_of_possible_mean": _mean([obs.recovery_fraction_of_possible for obs in group]),
                "hadamard_top_singular_mass_mean": _mean([obs.hadamard.top_singular_mass for obs in group]),
                "coupled_top_singular_mass_mean": _mean([obs.coupled.top_singular_mass for obs in group]),
                "near_rank1_fraction": sum(obs.hadamard_near_rank1 for obs in group) / len(group),
                "hadamard_spectrum_mean": [
                    _mean([obs.hadamard.singular_values[index] for obs in group])
                    for index in range(len(group[0].hadamard.singular_values))
                ],
                "coupled_spectrum_mean": [
                    _mean([obs.coupled.singular_values[index] for obs in group])
                    for index in range(len(group[0].coupled.singular_values))
                ],
            }
        )

    return {
        "observation_count": len(observations),
        "hadamard_effective_rank_mean": _mean(hadamard_ranks),
        "hadamard_effective_rank_stdev": _stdev(hadamard_ranks),
        "coupled_effective_rank_mean": _mean(coupled_ranks),
        "coupled_effective_rank_stdev": _stdev(coupled_ranks),
        "recovered_rank_mean": _mean(recovered),
        "recovered_rank_stdev": _stdev(recovered),
        "recovery_fraction_of_possible_mean": _mean(recovery_fractions),
        "hadamard_near_rank1_fraction": float(bottleneck_fraction),
        "by_layer_mode": by_layer_mode,
    }


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}"


def print_report(report: Mapping[str, Any], *, spectrum_top_k: int) -> None:
    summary = report["summary"]
    print("Rank bottleneck diagnostic")
    print(f"checkpoint: {report['checkpoint']}")
    print(
        "model: "
        f"coupling={report['model']['coupling_type']} "
        f"layers={report['model']['num_layers']} "
        f"state_shape={report['model']['state_shape']}"
    )
    print(f"probe: batch={report['probe']['batch_size']} seq_len={report['probe']['seq_len']} split={report['probe']['split']}")
    print()
    print("Global")
    print(
        "  hadamard effective rank: "
        f"{_format_float(summary['hadamard_effective_rank_mean'])} "
        f"+/- {_format_float(summary['hadamard_effective_rank_stdev'])}"
    )
    print(
        "  after coupling effective rank: "
        f"{_format_float(summary['coupled_effective_rank_mean'])} "
        f"+/- {_format_float(summary['coupled_effective_rank_stdev'])}"
    )
    print(
        "  rank recovered by coupling: "
        f"{_format_float(summary['recovered_rank_mean'])} "
        f"({_format_float(100.0 * summary['recovery_fraction_of_possible_mean'])}% of remaining possible rank)"
    )
    print(
        "  Hadamard near-rank-1 observations: "
        f"{_format_float(100.0 * summary['hadamard_near_rank1_fraction'])}%"
    )
    print(
        "  bottleneck verdict: "
        f"{'YES' if report['verdict']['hadamard_near_rank1_bottleneck'] else 'NO'}"
    )
    print()
    print("By layer/mode")
    for row in summary["by_layer_mode"]:
        hadamard_spectrum = row["hadamard_spectrum_mean"][:spectrum_top_k]
        coupled_spectrum = row["coupled_spectrum_mean"][:spectrum_top_k]
        print(
            "  "
            f"layer={row['layer']} mode={row['mode']} matrix={tuple(row['matrix_shape'])} "
            f"rank {row['hadamard_effective_rank_mean']:.3f} -> {row['coupled_effective_rank_mean']:.3f} "
            f"recovered={row['recovered_rank_mean']:.3f} "
            f"near_rank1={100.0 * row['near_rank1_fraction']:.1f}%"
        )
        print(f"    hadamard spectrum mean top-{len(hadamard_spectrum)}: {[round(v, 6) for v in hadamard_spectrum]}")
        print(f"    coupled spectrum mean top-{len(coupled_spectrum)}: {[round(v, 6) for v in coupled_spectrum]}")


def build_report(
    *,
    checkpoint: Path,
    loaded: LoadedCheckpoint,
    token_ids: Tensor,
    observations: Sequence[RankObservation],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = summarize_observations(observations)
    verdict = {
        "hadamard_near_rank1_bottleneck": bool(
            summary["hadamard_near_rank1_fraction"] >= args.bottleneck_fraction_threshold
        ),
        "bottleneck_fraction_threshold": float(args.bottleneck_fraction_threshold),
        "rank1_erank_threshold": float(args.rank1_erank_threshold),
        "rank1_top_mass_threshold": float(args.rank1_top_mass_threshold),
    }
    return {
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "warnings": loaded.warnings,
        "model": {
            "coupling_type": loaded.model.coupling_type,
            "num_layers": loaded.model.num_layers,
            "state_shape": list(loaded.model.state_shape),
            "normalization_type": loaded.model.normalization_type,
        },
        "probe": {
            "batch_size": int(token_ids.shape[0]),
            "seq_len": int(token_ids.shape[1]),
            "split": args.split if args.text is None and args.text_file is None else "provided_text",
            "seed": int(args.seed),
        },
        "summary": summary,
        "verdict": verdict,
        "observations": [asdict(obs) for obs in observations],
    }


def parse_modes(raw_modes: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if raw_modes is None:
        return None
    return tuple(raw_modes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze effective rank before and after Reciprocator coupling.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a training checkpoint.")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto.")
    parser.add_argument("--text", default=None, help="Optional literal probe text to encode and run.")
    parser.add_argument("--text-file", default=None, help="Optional probe text file.")
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Corpus split to sample when text is omitted.")
    parser.add_argument("--batch-size", type=int, default=None, help="Probe batch size; default min(checkpoint batch_size, 4).")
    parser.add_argument("--seq-len", type=int, default=None, help="Probe sequence length; default min(checkpoint seq_len, 128).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for corpus probe sampling.")
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        type=int,
        default=None,
        help="State tensor mode to unfold. Repeat to select multiple modes. Defaults to all modes.",
    )
    parser.add_argument(
        "--rank1-erank-threshold",
        type=float,
        default=1.25,
        help="Flag Hadamard unfoldings with effective rank at or below this value.",
    )
    parser.add_argument(
        "--rank1-top-mass-threshold",
        type=float,
        default=0.90,
        help="Flag Hadamard unfoldings where the top singular value carries this fraction of nuclear mass.",
    )
    parser.add_argument(
        "--bottleneck-fraction-threshold",
        type=float,
        default=0.50,
        help="Global verdict is YES when at least this fraction of observations are near-rank-1.",
    )
    parser.add_argument("--spectrum-top-k", type=int, default=8, help="Number of singular values to print per aggregate spectrum.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for the full JSON report.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.seq_len is not None and args.seq_len <= 0:
        raise ValueError("--seq-len must be positive.")
    if args.spectrum_top_k <= 0:
        raise ValueError("--spectrum-top-k must be positive.")

    device = resolve_device(args.device)
    loaded = load_trained_model(args.checkpoint, device=device)
    token_ids = build_probe_batch(loaded, args, device=device)
    modes = parse_modes(args.modes)

    _, _, observations = trace_rank_observations(
        loaded.model,
        token_ids,
        modes=modes,
        rank1_erank_threshold=args.rank1_erank_threshold,
        rank1_top_mass_threshold=args.rank1_top_mass_threshold,
    )
    report = build_report(
        checkpoint=args.checkpoint,
        loaded=loaded,
        token_ids=token_ids,
        observations=observations,
        args=args,
    )

    for warning in loaded.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print_report(report, spectrum_top_k=args.spectrum_top_k)

    if args.json_out is not None:
        output_path = args.json_out.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote JSON report: {output_path}")


if __name__ == "__main__":
    main()
