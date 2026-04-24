from __future__ import annotations

import math
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on some non-Unix platforms.
    resource = None

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from .corpora import read_corpus_text
from .rl.training import (
    LispGRPOConfig,
    RLStepMetrics,
    RLTrainingResult,
    SampleRecord,
    build_lisp_tokenizer,
    train_lisp_grpo,
)
from .rl.phase_monitor import PhaseTrajectoryMonitor
from .mixer import canonicalize_coupling_type, phase_aware_feature_map
from .model import LocalAttentionBlock, ReciprocatorLM

CHUNK_DRIFT_WARN_THRESHOLD = 0.05


@dataclass(frozen=True)
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text.")
        vocabulary = tuple(sorted(set(text)))
        stoi = {character: index for index, character in enumerate(vocabulary)}
        return cls(stoi=stoi, itos=vocabulary)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> Tensor:
        if not text:
            raise ValueError("Cannot encode empty text.")
        try:
            token_ids = [self.stoi[character] for character in text]
        except KeyError as exc:
            raise KeyError(f"Encountered unknown character during encoding: {exc.args[0]!r}") from exc
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.itos[int(token_id)] for token_id in token_ids)


@dataclass(frozen=True)
class TextDataset:
    tokenizer: CharTokenizer
    train_tokens: Tensor
    val_tokens: Tensor
    source_name: Optional[str] = None

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def token_frequencies(self) -> Tensor:
        counts = torch.bincount(self.train_tokens, minlength=self.vocab_size).to(torch.float32)
        return counts.clamp_min(1.0)


@dataclass(frozen=True)
class TrainingConfig:
    corpus_name: str = "greek_classics"
    max_chars: Optional[int] = 100000
    val_fraction: float = 0.1
    batch_size: int = 8
    seq_len: int = 128
    steps: int = 2000
    eval_every: int = 100
    eval_batches: int = 4
    learning_rate: float = 1e-3
    lr_warmup_steps: int = 0
    lr_decay_style: str = "constant"
    min_lr_scale: float = 0.1
    grad_clip_norm: Optional[float] = None
    weight_decay: float = 0.0
    hidden_size: int = 256
    state_shape: Tuple[int, ...] = (4, 4, 4)
    max_state_shape: Optional[Tuple[int, ...]] = None
    num_layers: int = 1
    ffn_expansion_factor: int = 2
    readout_type: str = "phase_aware"
    token_magnitude_type: str = "inverse_frequency_learned"
    phase_type: str = "rope"
    token_phase: str = "semantic"
    enable_self_relation: bool = True
    dynamic_gains: bool = False
    dynamic_spectral_gains: bool = False
    anisotropic_spectral_gains: bool = False
    gain_projector_rank: int = 8
    enable_cross_layer_state: bool = False
    coupling_type: str = "sequential"
    low_frequency_gain: float = 0.5
    low_frequency_sigma: float = 0.35
    high_frequency_gain: float = 0.5
    high_frequency_cutoff: float = 0.5
    wavelet_levels: Optional[int] = None
    normalization_type: str = "frobenius"
    device: str = "cpu"
    tensor_dynamic_growth: bool = True
    dynamic_mode_growth: bool = False
    dynamic_rank_growth: bool = False
    dynamic_mode_pruning: bool = False
    dynamic_rank_pruning: bool = False
    max_rank: Optional[int] = None
    growth_check_interval: int = 50
    growth_residual_threshold: float = 0.4
    post_growth_cooldown_checks: int = 0
    post_growth_cooldown_threshold_scale: float = 1.5
    residual_saturate_threshold: float = 0.4
    growth_residual_ema_decay: float = 0.95
    record_residual_diagnostics: bool = False
    chunk_size: Optional[int] = None
    track_chunk_drift: bool = False
    min_checks_before_first_growth: int = 0
    rank_growth_loss_ceiling: float = 1.5
    prune_threshold: float = 0.08
    prune_sustain_steps: int = 4
    prune_min_steps: int = 300
    mode_init: str = "zero"
    rank_init: str = "zero"
    generation_eval_samples: int = 0
    generation_prompt_len: int = 64
    generation_new_tokens: int = 128
    generation_temperature: float = 0.8
    generation_top_k: Optional[int] = 20
    benchmark_prompt_lengths: Tuple[int, ...] = ()
    benchmark_new_tokens: int = 128
    seed: int = 0
    stateful_training: bool = True
    block_layout: Optional[Tuple[str, ...]] = None
    attention_every_k: int = 0
    attention_num_heads: int = 8
    attention_window: int = 256
    attention_position: str = "after"


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    accuracy: float
    perplexity: float
    bpc: float


@dataclass(frozen=True)
class GenerationSample:
    prompt: str
    continuation: str
    full_text: str
    distinct_1: float
    distinct_2: float


@dataclass(frozen=True)
class RuntimeBenchmark:
    prompt_length: int
    prompt_wall_time_sec: float
    prompt_tokens_per_second: float
    decode_wall_time_sec: float
    decode_tokens_per_second: float
    peak_memory_bytes: Optional[int]
    memory_metric: str


@dataclass
class TrainingResult:
    config: TrainingConfig
    dataset: TextDataset
    model: ReciprocatorLM
    optimizer: Optimizer
    train_losses: list[float]
    train_metrics: list[Tuple[int, TrainingMetrics]]
    val_losses: list[Tuple[int, float]]
    val_metrics: list[Tuple[int, TrainingMetrics]]
    generation_samples: list[GenerationSample]
    runtime_benchmarks: list[RuntimeBenchmark]
    residual_diagnostics: list[dict]
    chunk_drift_history: list[dict]
    device: torch.device


@dataclass(frozen=True)
class TrainingResumeState:
    step: int
    model_state_dict: Dict[str, Tensor]
    optimizer_state_dict: Optional[dict]
    train_losses: Sequence[float] = ()
    val_losses: Sequence[Tuple[int, float]] = ()
    train_metrics: Sequence[Tuple[int, TrainingMetrics]] = ()
    val_metrics: Sequence[Tuple[int, TrainingMetrics]] = ()


TrainingStepCallback = Callable[
    [
        int,
        ReciprocatorLM,
        Optimizer,
        TextDataset,
        TrainingConfig,
        list[float],
        list[Tuple[int, float]],
        list[Tuple[int, TrainingMetrics]],
        list[Tuple[int, TrainingMetrics]],
        list[dict],
        torch.device,
    ],
    None,
]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _configure_tensor_dynamic_growth(device: torch.device, *, enabled: bool) -> None:
    if not enabled or device.type != "cuda":
        return

    cuda_memory = getattr(torch.cuda, "memory", None)
    configure_allocator = getattr(cuda_memory, "_set_allocator_settings", None)
    if configure_allocator is None:
        return

    try:
        configure_allocator("expandable_segments:True")
    except RuntimeError:
        # Some runtimes reject allocator changes after CUDA initialization.
        # Training can continue with the default allocator behavior.
        return


def _detect_state_shape_change(
    previous_shape: Tuple[int, ...],
    current_shape: Tuple[int, ...],
) -> Optional[Tuple[str, int]]:
    if previous_shape == current_shape:
        return None

    if len(current_shape) == len(previous_shape):
        changed_axes = [
            axis
            for axis, (previous_dim, current_dim) in enumerate(zip(previous_shape, current_shape))
            if previous_dim != current_dim
        ]
        if len(changed_axes) == 1:
            axis = changed_axes[0]
            if current_shape[axis] == previous_shape[axis] + 1:
                return "mode_growth", axis
            if current_shape[axis] == previous_shape[axis] - 1:
                return "mode_prune", axis
        return None

    if len(current_shape) == len(previous_shape) + 1 and current_shape[: len(previous_shape)] == previous_shape:
        return "rank_growth", len(previous_shape)

    if len(current_shape) == len(previous_shape) - 1:
        for axis in range(len(previous_shape)):
            if previous_shape[:axis] + previous_shape[axis + 1 :] == current_shape:
                return "rank_prune", axis

    return None


def _effective_max_state_shape(config: TrainingConfig) -> Tuple[int, ...]:
    current_shape = tuple(config.state_shape)
    if config.max_state_shape is None:
        return tuple(dim * 4 for dim in current_shape)

    max_shape = tuple(config.max_state_shape)
    if len(max_shape) >= len(current_shape):
        return max_shape[: len(current_shape)]
    return max_shape + tuple(dim * 4 for dim in current_shape[len(max_shape) :])


def _mode_growth_basis_rows(mixer, mode: int) -> Tensor:
    basis_rows = [
        mixer.decay_logit.detach().movedim(mode, 0).reshape(mixer.state_shape[mode], -1),
        mixer.input_logit.detach().movedim(mode, 0).reshape(mixer.state_shape[mode], -1),
        mixer.recurrent_logit.detach().movedim(mode, 0).reshape(mixer.state_shape[mode], -1),
    ]
    if mixer.self_relation_logit is not None:
        basis_rows.append(
            mixer.self_relation_logit.detach().movedim(mode, 0).reshape(mixer.state_shape[mode], -1)
        )
    return torch.cat(basis_rows, dim=0)


def _mode_signal_rows(state_signal: Tensor, mode: int) -> Tensor:
    state_shape = tuple(int(dim) for dim in state_signal.shape)
    return state_signal.abs().movedim(mode, 0).reshape(state_shape[mode], -1)


def _embed_mode_signal_rows(rows: Tensor, state_shape: Tuple[int, ...], mode: int) -> Tensor:
    other_shape = tuple(dim for axis, dim in enumerate(state_shape) if axis != mode)
    embedded_rows = []
    for index in range(state_shape[mode]):
        full_tensor = rows.new_zeros(state_shape)
        full_tensor.select(mode, index).copy_(rows[index].reshape(other_shape))
        embedded_rows.append(full_tensor.reshape(-1))
    return torch.stack(embedded_rows, dim=0)


def _row_space_residual_norm(basis_rows: Tensor, candidate: Tensor, *, eps: float = 1e-8) -> Tensor:
    candidate = candidate.to(basis_rows.dtype)
    nonzero_rows = basis_rows.norm(dim=1) > eps
    if not torch.any(nonzero_rows):
        return candidate.norm().to(torch.float32)

    basis_source = basis_rows[nonzero_rows].transpose(0, 1)
    left_singular_vectors, singular_values, _ = torch.linalg.svd(basis_source, full_matrices=False)
    rank = int((singular_values > eps).sum().item())
    if rank == 0:
        return candidate.norm().to(torch.float32)

    orthonormal_basis = left_singular_vectors[:, :rank]
    residual = candidate - orthonormal_basis @ (orthonormal_basis.mH @ candidate)
    return residual.norm().to(torch.float32)


def _compute_layer_mode_residual_norms(model: ReciprocatorLM, token_ids: Tensor) -> Tensor:
    layer_residual_norms = []
    model.eval()
    with torch.no_grad():
        hidden = model.token_lift(token_ids)
        for block in model.blocks:
            if not hasattr(block, "mixer"):
                hidden, _ = block(hidden, None)
                continue
            normalized_hidden = block.mixer.pre_norm(hidden)
            flat_hidden = normalized_hidden.reshape(-1, normalized_hidden.shape[-1])
            signal = block.mixer.signal_projector(flat_hidden).reshape(
                normalized_hidden.shape[0],
                normalized_hidden.shape[1],
                *block.mixer.state_shape,
            )
            state_signal = signal.mean(dim=(0, 1))

            block_norms = []
            for mode in range(len(block.mixer.state_shape)):
                candidate = state_signal.abs().mean(dim=mode).reshape(-1)
                basis_rows = _mode_growth_basis_rows(block.mixer, mode)
                block_norms.append(_row_space_residual_norm(basis_rows, candidate))

            block_norms_tensor = torch.stack(block_norms)
            layer_residual_norms.append(block_norms_tensor)

            hidden, _ = block(hidden, None)

    if not layer_residual_norms:
        raise ValueError("Cannot compute residual norms for a model with no blocks.")
    return torch.stack(layer_residual_norms, dim=0)


def _compute_mode_residual_norms(model: ReciprocatorLM, token_ids: Tensor) -> Tensor:
    return _compute_layer_mode_residual_norms(model, token_ids).mean(dim=0)


def _compute_mode_pruning_residual_norms(model: ReciprocatorLM, token_ids: Tensor) -> Tensor:
    residual_norms = None
    block_count = 0

    model.eval()
    with torch.no_grad():
        hidden = model.token_lift(token_ids)
        for block in model.blocks:
            if not hasattr(block, "mixer"):
                hidden, _ = block(hidden, None)
                continue
            normalized_hidden = block.mixer.pre_norm(hidden)
            flat_hidden = normalized_hidden.reshape(-1, normalized_hidden.shape[-1])
            signal = block.mixer.signal_projector(flat_hidden).reshape(
                normalized_hidden.shape[0],
                normalized_hidden.shape[1],
                *block.mixer.state_shape,
            )
            state_signal = signal.mean(dim=(0, 1))

            block_norms = []
            for mode in range(len(block.mixer.state_shape)):
                candidate_rows = _embed_mode_signal_rows(
                    _mode_signal_rows(state_signal, mode),
                    tuple(block.mixer.state_shape),
                    mode,
                )
                other_mode_rows = [
                    _embed_mode_signal_rows(
                        _mode_signal_rows(state_signal, other_mode),
                        tuple(block.mixer.state_shape),
                        other_mode,
                    )
                    for other_mode in range(len(block.mixer.state_shape))
                    if other_mode != mode
                ]
                if not other_mode_rows:
                    block_norms.append(candidate_rows.norm(dim=1).mean().to(torch.float32))
                    continue

                union_rows = torch.cat(other_mode_rows, dim=0)
                row_residuals = torch.stack(
                    [_row_space_residual_norm(union_rows, row) for row in candidate_rows],
                    dim=0,
                )
                block_norms.append(row_residuals.mean().to(torch.float32))

            block_norms_tensor = torch.stack(block_norms)
            if residual_norms is None:
                residual_norms = torch.zeros_like(block_norms_tensor)
            residual_norms = residual_norms + block_norms_tensor
            block_count += 1

            hidden, _ = block(hidden, None)

    if residual_norms is None or block_count == 0:
        raise ValueError("Cannot compute pruning residual norms for a model with no blocks.")
    return residual_norms / block_count


def _compute_mode_slice_activation_variances(
    model: ReciprocatorLM,
    token_ids: Tensor,
) -> list[Tensor]:
    slice_variances = None
    block_count = 0

    model.eval()
    with torch.no_grad():
        hidden = model.token_lift(token_ids)
        for block in model.blocks:
            if not hasattr(block, "mixer"):
                hidden, _ = block(hidden, None)
                continue
            normalized_hidden = block.mixer.pre_norm(hidden)
            flat_hidden = normalized_hidden.reshape(-1, normalized_hidden.shape[-1])
            signal = block.mixer.signal_projector(flat_hidden).reshape(
                normalized_hidden.shape[0],
                normalized_hidden.shape[1],
                *block.mixer.state_shape,
            )
            signal_magnitude = signal.abs()

            block_variances = []
            for mode in range(len(block.mixer.state_shape)):
                slice_rows = signal_magnitude.movedim(2 + mode, 0).reshape(block.mixer.state_shape[mode], -1)
                block_variances.append(slice_rows.var(dim=1, unbiased=False).to(torch.float32))

            if slice_variances is None:
                slice_variances = [torch.zeros_like(variances) for variances in block_variances]
            for mode, variances in enumerate(block_variances):
                slice_variances[mode] = slice_variances[mode] + variances
            block_count += 1

            hidden, _ = block(hidden, None)

    if slice_variances is None or block_count == 0:
        raise ValueError("Cannot compute mode slice activation variances for a model with no blocks.")
    return [variances / block_count for variances in slice_variances]


def _update_ema(previous: Optional[Tensor], current: Tensor, decay: float) -> Tensor:
    current = current.detach()
    if previous is None or previous.shape != current.shape:
        return current
    return previous * decay + current * (1.0 - decay)


def _update_tensor_list_ema(
    previous: Optional[Sequence[Tensor]],
    current: Sequence[Tensor],
    decay: float,
) -> list[Tensor]:
    if previous is None or len(previous) != len(current):
        return [tensor.detach() for tensor in current]
    if any(prev.shape != curr.shape for prev, curr in zip(previous, current)):
        return [tensor.detach() for tensor in current]
    return [
        prev.detach() * decay + curr.detach() * (1.0 - decay)
        for prev, curr in zip(previous, current)
    ]


def _update_pruning_candidate_streaks(
    previous_streaks: Sequence[int],
    smoothed_mode_pruning_residual_norms: Tensor,
    *,
    threshold: float,
) -> list[int]:
    current_rank = smoothed_mode_pruning_residual_norms.shape[0]
    if len(previous_streaks) != current_rank:
        return [0] * current_rank

    updated_streaks = []
    for mode in range(current_rank):
        if float(smoothed_mode_pruning_residual_norms[mode].item()) <= threshold:
            updated_streaks.append(int(previous_streaks[mode]) + 1)
        else:
            updated_streaks.append(0)
    return updated_streaks


def _effective_growth_residual_threshold(
    config: TrainingConfig,
    post_growth_cooldown_checks_remaining: int,
) -> float:
    threshold = float(config.growth_residual_threshold)
    if post_growth_cooldown_checks_remaining <= 0:
        return threshold
    return threshold * float(config.post_growth_cooldown_threshold_scale)


def _select_dynamic_growth_action(
    config: TrainingConfig,
    smoothed_mode_residual_norms: Tensor,
    recent_losses: Sequence[float],
    effective_growth_residual_threshold: Optional[float] = None,
) -> Optional[Tuple[str, int]]:
    current_shape = tuple(config.state_shape)
    current_rank = len(current_shape)
    if smoothed_mode_residual_norms.shape != (current_rank,):
        raise ValueError("smoothed_mode_residual_norms must have shape [rank].")

    threshold = (
        config.growth_residual_threshold
        if effective_growth_residual_threshold is None
        else effective_growth_residual_threshold
    )
    residual_saturate_threshold = config.residual_saturate_threshold
    max_rank = config.max_rank or current_rank + 2
    max_shape = _effective_max_state_shape(config)

    if config.dynamic_mode_growth:
        candidate_modes = [
            mode
            for mode, (current_dim, max_dim) in enumerate(zip(current_shape, max_shape))
            if current_dim < max_dim and float(smoothed_mode_residual_norms[mode].item()) > threshold
        ]
        if candidate_modes:
            grow_mode = max(candidate_modes, key=lambda mode: float(smoothed_mode_residual_norms[mode].item()))
            return "mode", grow_mode

    if config.dynamic_rank_growth and current_rank < max_rank:
        if torch.all(smoothed_mode_residual_norms <= residual_saturate_threshold).item() and recent_losses:
            loss_window = list(recent_losses[-config.growth_check_interval :])
            avg_loss = sum(loss_window) / len(loss_window)
            if avg_loss >= config.rank_growth_loss_ceiling:
                return "rank", current_rank

    return None


def _select_dynamic_mode_pruning_action(
    config: TrainingConfig,
    smoothed_mode_pruning_residual_norms: Tensor,
    prune_candidate_streaks: Sequence[int],
    mode_last_growth_steps: Sequence[int],
    axis_kinds: Sequence[str],
    *,
    step: int,
) -> Optional[int]:
    current_rank = len(config.state_shape)
    if not config.dynamic_mode_pruning:
        return None
    if smoothed_mode_pruning_residual_norms.shape != (current_rank,):
        raise ValueError("smoothed_mode_pruning_residual_norms must have shape [rank].")
    if len(prune_candidate_streaks) != current_rank:
        raise ValueError("prune_candidate_streaks must have one entry per mode.")
    if len(mode_last_growth_steps) != current_rank:
        raise ValueError("mode_last_growth_steps must have one entry per mode.")
    if len(axis_kinds) != current_rank:
        raise ValueError("axis_kinds must have one entry per mode.")

    threshold = config.prune_threshold
    required_streak = max(config.prune_sustain_steps, 1)
    candidate_modes = [
        mode
        for mode in range(current_rank)
        if axis_kinds[mode] == "mode"
        and int(config.state_shape[mode]) > 1
        and int(prune_candidate_streaks[mode]) >= required_streak
        and step - int(mode_last_growth_steps[mode]) > config.prune_min_steps
        and float(smoothed_mode_pruning_residual_norms[mode].item()) <= threshold
    ]
    if not candidate_modes:
        return None
    return min(candidate_modes, key=lambda mode: float(smoothed_mode_pruning_residual_norms[mode].item()))


def _select_dynamic_rank_pruning_action(
    config: TrainingConfig,
    smoothed_mode_pruning_residual_norms: Tensor,
    prune_candidate_streaks: Sequence[int],
    mode_last_growth_steps: Sequence[int],
    axis_kinds: Sequence[str],
    *,
    step: int,
) -> Optional[int]:
    current_rank = len(config.state_shape)
    if current_rank <= 1 or not config.dynamic_rank_pruning:
        return None
    if smoothed_mode_pruning_residual_norms.shape != (current_rank,):
        raise ValueError("smoothed_mode_pruning_residual_norms must have shape [rank].")
    if len(prune_candidate_streaks) != current_rank:
        raise ValueError("prune_candidate_streaks must have one entry per mode.")
    if len(mode_last_growth_steps) != current_rank:
        raise ValueError("mode_last_growth_steps must have one entry per mode.")
    if len(axis_kinds) != current_rank:
        raise ValueError("axis_kinds must have one entry per mode.")

    threshold = config.prune_threshold
    required_streak = max(config.prune_sustain_steps, 1)
    candidate_modes = [
        mode
        for mode in range(current_rank)
        if axis_kinds[mode] == "rank"
        and int(prune_candidate_streaks[mode]) >= required_streak
        and step - int(mode_last_growth_steps[mode]) > config.prune_min_steps
        and float(smoothed_mode_pruning_residual_norms[mode].item()) <= threshold
    ]
    if not candidate_modes:
        return None
    return min(candidate_modes, key=lambda mode: float(smoothed_mode_pruning_residual_norms[mode].item()))


def _select_mode_slice_to_prune(
    config: TrainingConfig,
    smoothed_mode_slice_activation_variances: Sequence[Tensor],
    *,
    pruned_axis: int,
) -> int:
    if pruned_axis >= len(smoothed_mode_slice_activation_variances):
        raise ValueError("smoothed_mode_slice_activation_variances must have one entry per mode.")
    slice_variances = smoothed_mode_slice_activation_variances[pruned_axis]
    expected_shape = (config.state_shape[pruned_axis],)
    if slice_variances.shape != expected_shape:
        raise ValueError(
            f"smoothed_mode_slice_activation_variances[{pruned_axis}] must have shape {expected_shape}."
        )
    return min(range(slice_variances.shape[0]), key=lambda index: float(slice_variances[index].item()))


def _set_optimizer_learning_rate_scale(optimizer: Optimizer, scale: float) -> None:
    for param_group in optimizer.param_groups:
        base_lr = param_group.setdefault("base_lr", param_group["lr"])
        param_group["lr"] = float(base_lr) * scale


def _learning_rate_scale_for_step(config: TrainingConfig, step: int) -> float:
    if step <= 0:
        raise ValueError("step must be positive.")

    warmup_steps = max(int(config.lr_warmup_steps), 0)
    min_lr_scale = float(config.min_lr_scale)

    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps

    if config.lr_decay_style == "constant":
        return 1.0

    decay_denominator = max(config.steps - warmup_steps, 1)
    decay_progress = min(max((step - warmup_steps) / decay_denominator, 0.0), 1.0)
    if config.lr_decay_style == "linear":
        return min_lr_scale + (1.0 - min_lr_scale) * (1.0 - decay_progress)
    if config.lr_decay_style == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine
    raise ValueError(f"Unsupported lr_decay_style: {config.lr_decay_style!r}")


def _build_metrics(loss: float, accuracy: float) -> TrainingMetrics:
    return TrainingMetrics(
        loss=loss,
        accuracy=accuracy,
        perplexity=math.exp(loss),
        bpc=loss / math.log(2.0),
    )


def _distinct_ngram_ratio(token_ids: Sequence[int], n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive.")
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def _sample_next_token(
    logits: Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("Expected logits to have shape [batch, vocab_size].")

    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < scaled_logits.shape[-1]:
        top_values, top_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
        probabilities = torch.softmax(top_values, dim=-1)
        sampled_index = torch.multinomial(probabilities, num_samples=1)
        return top_indices.gather(-1, sampled_index)

    probabilities = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def _generate_continuation_tokens(
    model: ReciprocatorLM,
    prompt_tokens: Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
) -> list[int]:
    if prompt_tokens.ndim != 1:
        raise ValueError("prompt_tokens must have shape [prompt_len].")
    if prompt_tokens.numel() == 0:
        raise ValueError("prompt_tokens must contain at least one token.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if max_new_tokens == 0:
        return []

    prompt_batch = prompt_tokens.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, states = model(prompt_batch, position_offset=0)
        current_logits = logits[:, -1]
        current_position = int(prompt_batch.shape[1])
        generated_tokens: list[int] = []

        for generation_step in range(max_new_tokens):
            next_token = _sample_next_token(
                current_logits,
                temperature=temperature,
                top_k=top_k,
            )
            generated_tokens.append(int(next_token.item()))
            if generation_step == max_new_tokens - 1:
                break

            logits, states = model(
                next_token,
                states=states,
                position_offset=current_position,
            )
            current_logits = logits[:, -1]
            current_position += 1

    return generated_tokens


def _select_prompt_tokens(
    tokens: Tensor,
    *,
    prompt_len: int,
    index: int = 0,
    total: int = 1,
) -> Tensor:
    if tokens.ndim != 1:
        raise ValueError("Expected tokens to have shape [num_tokens].")
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive.")
    if tokens.numel() <= prompt_len:
        raise ValueError(
            f"Need more than prompt_len={prompt_len} tokens to choose a prompt, "
            f"but received {tokens.numel()}."
        )

    max_start = max(int(tokens.numel() - prompt_len - 1), 0)
    if total <= 1 or max_start == 0:
        start = 0
    else:
        start = round(max_start * (index / (total - 1)))
    return tokens[start : start + prompt_len].clone()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _process_peak_memory_bytes() -> Optional[int]:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        usage *= 1024
    return int(usage)


def _peak_memory_bytes(device: torch.device) -> Tuple[Optional[int], str]:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device)), "cuda_max_memory_allocated"
    return _process_peak_memory_bytes(), "process_peak_rss"


def evaluate_generation_samples(
    model: ReciprocatorLM,
    dataset: TextDataset,
    *,
    prompt_len: int,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
) -> list[GenerationSample]:
    if num_samples <= 0:
        return []
    if prompt_len <= 0:
        raise ValueError("generation_prompt_len must be positive.")
    if max_new_tokens < 0:
        raise ValueError("generation_new_tokens must be non-negative.")

    source_tokens = dataset.val_tokens
    if source_tokens.numel() <= prompt_len:
        source_tokens = dataset.train_tokens

    samples: list[GenerationSample] = []
    for index in range(num_samples):
        prompt_tokens = _select_prompt_tokens(
            source_tokens,
            prompt_len=prompt_len,
            index=index,
            total=num_samples,
        )
        continuation_tokens = _generate_continuation_tokens(
            model,
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        prompt_text = dataset.tokenizer.decode(prompt_tokens.tolist())
        continuation_text = dataset.tokenizer.decode(continuation_tokens) if continuation_tokens else ""
        samples.append(
            GenerationSample(
                prompt=prompt_text,
                continuation=continuation_text,
                full_text=prompt_text + continuation_text,
                distinct_1=_distinct_ngram_ratio(continuation_tokens, 1),
                distinct_2=_distinct_ngram_ratio(continuation_tokens, 2),
            )
        )

    return samples


def benchmark_streaming_inference(
    model: ReciprocatorLM,
    dataset: TextDataset,
    *,
    prompt_lengths: Sequence[int],
    decode_tokens: int,
    device: torch.device,
) -> list[RuntimeBenchmark]:
    unique_prompt_lengths = sorted({int(length) for length in prompt_lengths if int(length) > 0})
    if not unique_prompt_lengths:
        return []
    if decode_tokens <= 0:
        raise ValueError("benchmark_new_tokens must be positive when benchmarking is enabled.")

    source_tokens = dataset.val_tokens
    if source_tokens.numel() <= max(unique_prompt_lengths) + 1:
        source_tokens = dataset.train_tokens

    model.eval()
    benchmarks: list[RuntimeBenchmark] = []
    with torch.no_grad():
        for prompt_length in unique_prompt_lengths:
            prompt_tokens = _select_prompt_tokens(source_tokens, prompt_len=prompt_length)
            prompt_batch = prompt_tokens.unsqueeze(0).to(device)

            _reset_peak_memory(device)
            _synchronize_device(device)
            prompt_start = time.perf_counter()
            logits, states = model(prompt_batch, position_offset=0)
            _synchronize_device(device)
            prompt_elapsed = time.perf_counter() - prompt_start

            current_logits = logits[:, -1]
            current_position = prompt_length
            _synchronize_device(device)
            decode_start = time.perf_counter()
            for decode_step in range(decode_tokens):
                next_token = current_logits.argmax(dim=-1, keepdim=True)
                if decode_step == decode_tokens - 1:
                    break
                logits, states = model(
                    next_token,
                    states=states,
                    position_offset=current_position,
                )
                current_logits = logits[:, -1]
                current_position += 1
            _synchronize_device(device)
            decode_elapsed = time.perf_counter() - decode_start
            peak_memory_bytes, memory_metric = _peak_memory_bytes(device)

            benchmarks.append(
                RuntimeBenchmark(
                    prompt_length=prompt_length,
                    prompt_wall_time_sec=prompt_elapsed,
                    prompt_tokens_per_second=(prompt_length / prompt_elapsed) if prompt_elapsed > 0.0 else float("inf"),
                    decode_wall_time_sec=decode_elapsed,
                    decode_tokens_per_second=(decode_tokens / decode_elapsed) if decode_elapsed > 0.0 else float("inf"),
                    peak_memory_bytes=peak_memory_bytes,
                    memory_metric=memory_metric,
                )
            )

    return benchmarks


def _truncate_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None:
        return text
    if max_chars <= 0:
        raise ValueError("max_chars must be positive when provided.")
    return text[:max_chars]


def build_text_dataset(
    text: str,
    *,
    source_name: Optional[str] = None,
    max_chars: Optional[int] = None,
    val_fraction: float = 0.1,
    min_split_tokens: int = 32,
    tokenizer: Optional[CharTokenizer] = None,
) -> TextDataset:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    if min_split_tokens < 2:
        raise ValueError("min_split_tokens must be at least 2.")

    truncated = _truncate_text(text, max_chars=max_chars)
    if not truncated:
        raise ValueError("Text is empty after applying max_chars.")

    if tokenizer is None:
        tokenizer = CharTokenizer.from_text(truncated)
    encoded = tokenizer.encode(truncated)

    min_total_tokens = max(2 * min_split_tokens, 4)
    if encoded.numel() < min_total_tokens:
        raise ValueError(
            f"Need at least {min_total_tokens} characters to build a train/val split, "
            f"but received {encoded.numel()}."
        )

    split_index = int(encoded.numel() * (1.0 - val_fraction))
    split_index = min(max(split_index, min_split_tokens), encoded.numel() - min_split_tokens)

    train_tokens = encoded[:split_index].clone()
    val_tokens = encoded[split_index:].clone()
    return TextDataset(
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        source_name=source_name,
    )


def build_corpus_dataset(
    corpus_name: str,
    *,
    max_chars: Optional[int] = None,
    val_fraction: float = 0.1,
    min_split_tokens: int = 32,
    tokenizer: Optional[CharTokenizer] = None,
) -> TextDataset:
    return build_text_dataset(
        read_corpus_text(corpus_name),
        source_name=corpus_name,
        max_chars=max_chars,
        val_fraction=val_fraction,
        min_split_tokens=min_split_tokens,
        tokenizer=tokenizer,
    )


def sample_causal_lm_batch(
    tokens: Tensor,
    batch_size: int,
    seq_len: int,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    if tokens.ndim != 1:
        raise ValueError("Expected tokens to have shape [num_tokens].")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if tokens.numel() <= seq_len:
        raise ValueError(
            f"Need more than seq_len={seq_len} tokens to sample a batch, "
            f"but received {tokens.numel()}."
        )

    max_start = tokens.numel() - seq_len
    starts = torch.randint(0, max_start, (batch_size,))
    inputs = torch.stack([tokens[start : start + seq_len] for start in starts], dim=0)
    targets = torch.stack([tokens[start + 1 : start + seq_len + 1] for start in starts], dim=0)

    if device is not None:
        inputs = inputs.to(device)
        targets = targets.to(device)
    return inputs, targets


def _flatten_complex_features(hidden: Tensor) -> Tensor:
    return torch.cat([hidden.real, hidden.imag], dim=-1)


def _collect_growth_reference_contexts(
    model: ReciprocatorLM,
    token_ids: Tensor,
) -> list[dict[str, Tensor]]:
    contexts: list[dict[str, Tensor]] = []
    model.eval()
    with torch.no_grad():
        hidden = model.token_lift(token_ids)
        for block in model.blocks:
            if not hasattr(block, "mixer"):
                hidden, _ = block(hidden, None)
                continue
            normalized_hidden = block.mixer.pre_norm(hidden)
            flat_hidden = normalized_hidden.reshape(-1, normalized_hidden.shape[-1])
            signal = block.mixer.signal_projector(flat_hidden).reshape(
                normalized_hidden.shape[0],
                normalized_hidden.shape[1],
                *block.mixer.state_shape,
            )
            contexts.append(
                {
                    "hidden_features": _flatten_complex_features(normalized_hidden).mean(dim=(0, 1)),
                    "state_signal": signal.mean(dim=(0, 1)),
                    "state_signal_features": _state_signal_features(signal.mean(dim=(0, 1))),
                }
            )
            hidden, _ = block(hidden, None)
    return contexts


def _state_signal_features(state_signal: Tensor) -> Tensor:
    return phase_aware_feature_map(state_signal, batch_dim=False)


def _collect_rank_growth_reference_contexts(
    model: ReciprocatorLM,
    token_ids: Tensor,
) -> list[dict[str, Tensor]]:
    contexts: list[dict[str, Tensor]] = []
    model.eval()
    with torch.no_grad():
        hidden = model.token_lift(token_ids)
        for block in model.blocks:
            next_hidden_steps = []
            hidden_residual_steps = []
            residual_signal_steps = []
            state = None
            for position in range(hidden.shape[1]):
                hidden_t = hidden[:, position]
                normalized_hidden_t = block.mixer.pre_norm(hidden_t)
                delta_t, state = block.mixer.step(hidden_t, state)
                hidden_residual_t = normalized_hidden_t - delta_t
                hidden_residual_steps.append(hidden_residual_t)
                residual_signal_steps.append(block.mixer.signal_projector(hidden_residual_t))
                next_hidden_steps.append(hidden_t + block.ffn(block.ffn_norm(delta_t)))

            hidden_residual = torch.stack(hidden_residual_steps, dim=1)
            rank_residual_state_signal = torch.stack(residual_signal_steps, dim=1).mean(dim=(0, 1))
            contexts.append(
                {
                    "hidden_residual_features": _flatten_complex_features(hidden_residual).mean(dim=(0, 1)),
                    "rank_residual_state_signal": rank_residual_state_signal,
                    "rank_residual_state_features": _state_signal_features(rank_residual_state_signal),
                }
            )
            hidden = torch.stack(next_hidden_steps, dim=1)
    return contexts


def _orthogonalize_candidate(
    old_rows: Tensor,
    candidate: Tensor,
    fallback: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    basis_source = old_rows.transpose(0, 1)
    q, _ = torch.linalg.qr(basis_source, mode="reduced")

    candidate = candidate.to(old_rows.dtype)
    fallback = fallback.to(old_rows.dtype)

    residual = candidate - q @ (q.mH @ candidate)
    if residual.norm() < eps:
        residual = fallback - q @ (q.mH @ fallback)
    if residual.norm() < eps:
        residual = fallback

    target_norm = old_rows.norm(dim=1).mean().clamp_min(eps)
    return residual / residual.norm().clamp_min(eps) * target_norm


def _scale_candidate_to_match_rows(
    old_rows: Tensor,
    candidate: Tensor,
    fallback: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    candidate = candidate.to(old_rows.dtype)
    fallback = fallback.to(old_rows.dtype)
    if candidate.norm() < eps:
        candidate = fallback
    if candidate.norm() < eps:
        return fallback
    target_norm = old_rows.norm(dim=1).mean().clamp_min(eps)
    return candidate / candidate.norm().clamp_min(eps) * target_norm


_BLOCK_MIXER_KEY_RE = re.compile(r"^blocks\.(\d+)\.mixer\.(.+)$")
_MODE_WEIGHT_KEY_RE = re.compile(r"^coupling\.mode_weights\.(\d+)$")


@dataclass(frozen=True)
class _StateGrowthLayout:
    old_state_shape: Tuple[int, ...]
    new_state_shape: Tuple[int, ...]
    growth_kind: str
    grown_axis: int


def _shape_tuple(shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _assert_tensor_can_view_as(
    tensor: Tensor,
    view_shape: Tuple[int, ...],
    *,
    key: str,
    role: str,
) -> None:
    expected_numel = math.prod(view_shape)
    actual_numel = int(tensor.numel())
    if actual_numel != expected_numel:
        raise ValueError(
            f"{key}: cannot view {role} tensor with shape {_shape_tuple(tensor.shape)} "
            f"as {view_shape}; expected {expected_numel} elements, got {actual_numel}."
        )


def _assert_tensor_numel_matches(
    tensor: Tensor,
    reference: Tensor,
    *,
    key: str,
    role: str,
) -> None:
    if tensor.numel() != reference.numel():
        raise ValueError(
            f"{key}: {role} has shape {_shape_tuple(tensor.shape)} but target slice "
            f"has shape {_shape_tuple(reference.shape)}."
        )


def _copy_overlapping_tensor(old_param: Tensor, new_param: Tensor) -> Tensor:
    grown = torch.zeros_like(new_param)
    common_rank = min(old_param.ndim, new_param.ndim)
    common_slices = tuple(
        slice(0, min(old_param.shape[index], new_param.shape[index]))
        for index in range(common_rank)
    )
    target_index = common_slices + (0,) * (new_param.ndim - common_rank)
    source_index = common_slices + (0,) * (old_param.ndim - common_rank)
    grown[target_index] = old_param[source_index]
    return grown


def _copy_with_grown_state_layout(
    key: str,
    old_param: Tensor,
    new_param: Tensor,
    *,
    layout: _StateGrowthLayout,
    prefix_shape: Tuple[int, ...],
    suffix_shape: Tuple[int, ...],
    init_strategy: str,
    orthogonal_reference_slice: Optional[Tensor] = None,
    residual_reference_slice: Optional[Tensor] = None,
) -> Tensor:
    old_view_shape = (*prefix_shape, *layout.old_state_shape, *suffix_shape)
    new_view_shape = (*prefix_shape, *layout.new_state_shape, *suffix_shape)
    _assert_tensor_can_view_as(old_param, old_view_shape, key=key, role="old")
    _assert_tensor_can_view_as(new_param, new_view_shape, key=key, role="new")

    old_view = old_param.reshape(*old_view_shape)
    fresh_view = new_param.reshape(*new_view_shape).clone()
    new_view = torch.zeros_like(fresh_view)
    prefix_rank = len(prefix_shape)
    suffix_rank = len(suffix_shape)

    if layout.growth_kind == "mode":
        copy_index = [slice(None)] * (prefix_rank + len(layout.old_state_shape) + suffix_rank)
        copy_index[prefix_rank + layout.grown_axis] = slice(0, layout.old_state_shape[layout.grown_axis])
        new_view[tuple(copy_index)] = old_view

        new_slice_index = list(copy_index)
        new_slice_index[prefix_rank + layout.grown_axis] = layout.old_state_shape[layout.grown_axis]
        new_slice = new_view[tuple(new_slice_index)]
        fresh_slice = fresh_view[tuple(new_slice_index)]

        if init_strategy == "mean":
            new_slice.copy_(old_view.mean(dim=prefix_rank + layout.grown_axis))
        elif init_strategy == "orthogonal":
            candidate = orthogonal_reference_slice
            if candidate is None:
                candidate = fresh_slice
            _assert_tensor_numel_matches(
                candidate,
                fresh_slice,
                key=key,
                role="orthogonal reference slice",
            )
            old_rows = old_view.movedim(
                prefix_rank + layout.grown_axis,
                0,
            ).reshape(layout.old_state_shape[layout.grown_axis], -1)
            fill = _orthogonalize_candidate(
                old_rows,
                candidate.reshape(-1),
                fresh_slice.reshape(-1),
            )
            new_slice.copy_(fill.reshape_as(fresh_slice).to(new_slice.dtype))
        elif init_strategy == "residual":
            candidate = residual_reference_slice
            if candidate is None:
                candidate = fresh_slice
            _assert_tensor_numel_matches(
                candidate,
                fresh_slice,
                key=key,
                role="residual reference slice",
            )
            old_rows = old_view.movedim(
                prefix_rank + layout.grown_axis,
                0,
            ).reshape(layout.old_state_shape[layout.grown_axis], -1)
            fill = _scale_candidate_to_match_rows(
                old_rows,
                candidate.reshape(-1),
                fresh_slice.reshape(-1),
            )
            new_slice.copy_(fill.reshape_as(fresh_slice).to(new_slice.dtype))
    else:
        copy_index = (
            [slice(None)] * prefix_rank
            + [slice(None)] * len(layout.old_state_shape)
            + [0]
            + [slice(None)] * suffix_rank
        )
        new_view[tuple(copy_index)] = old_view

        new_slice_index = (
            [slice(None)] * prefix_rank
            + [slice(None)] * len(layout.old_state_shape)
            + [1]
            + [slice(None)] * suffix_rank
        )
        new_slice = new_view[tuple(new_slice_index)]
        if init_strategy == "mean":
            state_axes = tuple(range(prefix_rank, prefix_rank + len(layout.old_state_shape)))
            fill = old_view.mean(dim=state_axes, keepdim=True)
            new_slice.copy_(fill.expand_as(new_slice))
        elif init_strategy == "orthogonal":
            candidate = orthogonal_reference_slice
            if candidate is None:
                candidate = fresh_view[tuple(new_slice_index)]
            _assert_tensor_numel_matches(
                candidate,
                new_slice,
                key=key,
                role="orthogonal reference slice",
            )
            old_rows = old_view.reshape(1, -1)
            fill = _orthogonalize_candidate(
                old_rows,
                candidate.reshape(-1),
                fresh_view[tuple(new_slice_index)].reshape(-1),
            )
            new_slice.copy_(fill.reshape_as(new_slice).to(new_slice.dtype))
        elif init_strategy == "residual":
            candidate = residual_reference_slice
            if candidate is None:
                candidate = fresh_view[tuple(new_slice_index)]
            _assert_tensor_numel_matches(
                candidate,
                new_slice,
                key=key,
                role="residual reference slice",
            )
            old_rows = old_view.reshape(1, -1)
            fill = _scale_candidate_to_match_rows(
                old_rows,
                candidate.reshape(-1),
                fresh_view[tuple(new_slice_index)].reshape(-1),
            )
            new_slice.copy_(fill.reshape_as(new_slice).to(new_slice.dtype))

    return new_view.reshape_as(new_param)


def _copy_mode_weight_matrix_for_growth(
    old_sd: Dict[str, Tensor],
    old_param: Optional[Tensor],
    new_param: Tensor,
    *,
    growth_kind: str,
    init_strategy: str,
    block_index: int,
) -> Tensor:
    if growth_kind == "rank" and old_param is None:
        if init_strategy == "zero":
            return torch.zeros_like(new_param)
        means = [
            tensor.mean()
            for key, tensor in old_sd.items()
            if key.startswith(f"blocks.{block_index}.mixer.coupling.mode_weights.")
        ]
        if not means:
            return new_param
        mean_value = torch.stack(means).mean().to(new_param.dtype)
        if init_strategy in {"orthogonal", "residual"}:
            seeded = torch.zeros_like(new_param)
            seeded.diagonal().fill_(mean_value)
            return seeded
        return torch.full_like(new_param, mean_value)

    assert old_param is not None
    if old_param.shape == new_param.shape:
        return old_param

    grown = torch.zeros_like(new_param)
    grown[: old_param.shape[0], : old_param.shape[1]] = old_param
    if init_strategy == "mean":
        grown[-1, : old_param.shape[1]] = old_param.mean(dim=0)
        grown[: old_param.shape[0], -1] = old_param.mean(dim=1)
        grown[-1, -1] = old_param.diagonal().mean()
    elif init_strategy in {"orthogonal", "residual"}:
        grown[-1, -1] = old_param.diagonal().mean()
    return grown


def _pad_copy_state_dict(
    old_sd: Dict[str, Tensor],
    new_sd: Dict[str, Tensor],
    *,
    old_state_shape: Optional[Tuple[int, ...]] = None,
    new_state_shape: Optional[Tuple[int, ...]] = None,
    growth_kind: Optional[str] = None,
    grown_axis: Optional[int] = None,
    mode_init: str = "zero",
    rank_init: str = "zero",
    reference_contexts: Optional[list[dict[str, Tensor]]] = None,
) -> Dict[str, Tensor]:
    if (
        old_state_shape is None
        or new_state_shape is None
        or growth_kind is None
        or grown_axis is None
    ):
        merged = {}
        for key, new_param in new_sd.items():
            if key not in old_sd:
                merged[key] = new_param
                continue
            old_param = old_sd[key]
            if old_param.shape == new_param.shape:
                merged[key] = old_param
                continue
            merged[key] = _copy_overlapping_tensor(old_param, new_param)
        return merged

    layout = _StateGrowthLayout(
        old_state_shape=old_state_shape,
        new_state_shape=new_state_shape,
        growth_kind=growth_kind,
        grown_axis=grown_axis,
    )
    merged = {}
    for key, new_param in new_sd.items():
        block_match = _BLOCK_MIXER_KEY_RE.match(key)
        if key not in old_sd:
            if growth_kind == "rank" and block_match is not None:
                block_index = int(block_match.group(1))
                suffix = block_match.group(2)
                if _MODE_WEIGHT_KEY_RE.match(suffix):
                    merged[key] = _copy_mode_weight_matrix_for_growth(
                        old_sd,
                        None,
                        new_param,
                        growth_kind=growth_kind,
                        init_strategy=rank_init,
                        block_index=block_index,
                    )
                    continue
            merged[key] = new_param
            continue

        old_param = old_sd[key]
        if old_param.shape == new_param.shape:
            merged[key] = old_param
            continue

        init_strategy = mode_init if growth_kind == "mode" else rank_init
        if key.startswith("cross_layer_proj.") and key.endswith(".weight"):
            feature_channels = _cross_layer_feature_channels(old_param, old_state_shape)
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(old_param.shape[0], feature_channels),
                suffix_shape=(),
                init_strategy="zero",
            )
            continue

        if block_match is None:
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(),
                suffix_shape=(),
                init_strategy=init_strategy,
            )
            continue

        block_index = int(block_match.group(1))
        suffix = block_match.group(2)
        reference_context = None if reference_contexts is None else reference_contexts[block_index]

        if suffix in {"decay_logit", "input_logit", "recurrent_logit", "self_relation_logit"}:
            orthogonal_reference_slice = None
            residual_reference_slice = None
            if init_strategy == "orthogonal" and reference_context is not None:
                if growth_kind == "mode":
                    orthogonal_reference_slice = reference_context["state_signal"].abs().mean(dim=grown_axis)
                else:
                    orthogonal_reference_slice = reference_context["state_signal"].abs()
            elif init_strategy == "residual" and reference_context is not None:
                if growth_kind == "mode":
                    residual_reference_slice = reference_context["state_signal"].abs().mean(dim=grown_axis)
                else:
                    residual_reference_slice = reference_context["rank_residual_state_signal"].abs()
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(),
                suffix_shape=(),
                init_strategy=init_strategy,
                orthogonal_reference_slice=orthogonal_reference_slice,
                residual_reference_slice=residual_reference_slice,
            )
            continue

        if suffix in {"signal_projector.magnitude_proj.bias", "signal_projector.phase_proj.bias"}:
            orthogonal_reference_slice = None
            residual_reference_slice = None
            if init_strategy == "orthogonal" and reference_context is not None:
                if growth_kind == "mode":
                    orthogonal_reference_slice = reference_context["state_signal"].abs().mean(dim=grown_axis)
                else:
                    orthogonal_reference_slice = reference_context["state_signal"].abs()
            elif init_strategy == "residual" and reference_context is not None:
                if growth_kind == "mode":
                    residual_reference_slice = reference_context["state_signal"].abs().mean(dim=grown_axis)
                else:
                    residual_reference_slice = reference_context["rank_residual_state_signal"].abs()
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(),
                suffix_shape=(),
                init_strategy=init_strategy,
                orthogonal_reference_slice=orthogonal_reference_slice,
                residual_reference_slice=residual_reference_slice,
            )
            continue

        if suffix in {"signal_projector.magnitude_proj.weight", "signal_projector.phase_proj.weight"}:
            orthogonal_reference_slice = None
            residual_reference_slice = None
            if init_strategy == "orthogonal" and reference_context is not None:
                reference = reference_context["hidden_features"]
                if growth_kind == "mode":
                    expand_shape = (
                        *new_state_shape[:grown_axis],
                        *new_state_shape[grown_axis + 1 :],
                        reference.shape[0],
                    )
                else:
                    expand_shape = (*old_state_shape, reference.shape[0])
                orthogonal_reference_slice = reference.reshape(
                    *([1] * (len(expand_shape) - 1)),
                    reference.shape[0],
                ).expand(expand_shape)
            elif init_strategy == "residual" and reference_context is not None:
                if growth_kind == "mode":
                    reference = reference_context["hidden_features"]
                    residual_signal = reference_context["state_signal"].abs().mean(dim=grown_axis)
                else:
                    reference = reference_context["hidden_residual_features"]
                    residual_signal = reference_context["rank_residual_state_signal"].abs()
                residual_reference_slice = (
                    residual_signal.unsqueeze(-1)
                    * reference.reshape(*([1] * residual_signal.ndim), reference.shape[0])
                )
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(),
                suffix_shape=(old_param.shape[1],),
                init_strategy=init_strategy,
                orthogonal_reference_slice=orthogonal_reference_slice,
                residual_reference_slice=residual_reference_slice,
            )
            continue

        if suffix == "gain_projector.0.weight":
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(old_param.shape[0], 2),
                suffix_shape=(),
                init_strategy="zero",
            )
            continue

        if suffix == "gain_projector.2.weight":
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(3,),
                suffix_shape=(old_param.shape[1],),
                init_strategy="zero",
            )
            continue

        if suffix == "gain_projector.2.bias":
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(3,),
                suffix_shape=(),
                init_strategy="zero",
            )
            continue

        if suffix == "return_map.proj.weight":
            effective_init = "mean" if init_strategy == "orthogonal" else init_strategy
            residual_reference_slice = None
            feature_channels = _return_map_feature_channels(old_param, old_state_shape)
            if init_strategy == "residual" and reference_context is not None:
                if growth_kind == "mode":
                    signal_features = reference_context["state_signal_features"].mean(dim=1 + grown_axis)
                    feature_source = reference_context["hidden_features"]
                else:
                    signal_features = reference_context["rank_residual_state_features"]
                    feature_source = reference_context["hidden_residual_features"]
                residual_reference_slice = (
                    feature_source.reshape(
                        old_param.shape[0],
                        *([1] * signal_features.ndim),
                    )
                    * signal_features.unsqueeze(0)
                )
            merged[key] = _copy_with_grown_state_layout(
                key,
                old_param,
                new_param,
                layout=layout,
                prefix_shape=(old_param.shape[0], feature_channels),
                suffix_shape=(),
                init_strategy=effective_init,
                residual_reference_slice=residual_reference_slice,
            )
            continue

        mode_weight_match = _MODE_WEIGHT_KEY_RE.match(suffix)
        if mode_weight_match is not None:
            mode_index = int(mode_weight_match.group(1))
            if growth_kind == "mode" and mode_index == grown_axis:
                merged[key] = _copy_mode_weight_matrix_for_growth(
                    old_sd,
                    old_param,
                    new_param,
                    growth_kind=growth_kind,
                    init_strategy=init_strategy,
                    block_index=block_index,
                )
            else:
                merged[key] = old_param
            continue

        merged[key] = _copy_overlapping_tensor(old_param, new_param)
    return merged


def _prune_copy_state_dict(
    old_sd: Dict[str, Tensor],
    new_sd: Dict[str, Tensor],
    *,
    old_state_shape: Tuple[int, ...],
    new_state_shape: Tuple[int, ...],
    pruned_axis: int,
) -> Dict[str, Tensor]:
    def reduce_state_layout(
        old_param: Tensor,
        new_param: Tensor,
        *,
        prefix_shape: Tuple[int, ...],
        suffix_shape: Tuple[int, ...],
    ) -> Tensor:
        old_view = old_param.reshape(*prefix_shape, *old_state_shape, *suffix_shape)
        reduced = old_view.mean(dim=len(prefix_shape) + pruned_axis)
        return reduced.reshape_as(new_param).to(new_param.dtype)

    merged = {}
    for key, new_param in new_sd.items():
        block_match = _BLOCK_MIXER_KEY_RE.match(key)
        mapped_key = key
        suffix = None

        if block_match is not None:
            block_index = int(block_match.group(1))
            suffix = block_match.group(2)
            mode_weight_match = _MODE_WEIGHT_KEY_RE.match(suffix)
            if mode_weight_match is not None:
                new_mode_index = int(mode_weight_match.group(1))
                old_mode_index = new_mode_index if new_mode_index < pruned_axis else new_mode_index + 1
                mapped_key = f"blocks.{block_index}.mixer.coupling.mode_weights.{old_mode_index}"

        if mapped_key not in old_sd:
            merged[key] = new_param
            continue

        old_param = old_sd[mapped_key]
        if old_param.shape == new_param.shape:
            merged[key] = old_param
            continue

        if key.startswith("cross_layer_proj.") and key.endswith(".weight"):
            feature_channels = _cross_layer_feature_channels(old_param, old_state_shape)
            merged[key] = reduce_state_layout(
                old_param,
                new_param,
                prefix_shape=(old_param.shape[0], feature_channels),
                suffix_shape=(),
            )
            continue

        if block_match is None:
            merged[key] = new_param
            continue

        if suffix in {"decay_logit", "input_logit", "recurrent_logit", "self_relation_logit"}:
            merged[key] = reduce_state_layout(old_param, new_param, prefix_shape=(), suffix_shape=())
            continue

        if suffix in {"signal_projector.magnitude_proj.bias", "signal_projector.phase_proj.bias"}:
            merged[key] = reduce_state_layout(old_param, new_param, prefix_shape=(), suffix_shape=())
            continue

        if suffix in {"signal_projector.magnitude_proj.weight", "signal_projector.phase_proj.weight"}:
            merged[key] = reduce_state_layout(
                old_param,
                new_param,
                prefix_shape=(),
                suffix_shape=(old_param.shape[1],),
            )
            continue

        if suffix == "gain_projector.0.weight":
            merged[key] = reduce_state_layout(
                old_param,
                new_param,
                prefix_shape=(old_param.shape[0], 2),
                suffix_shape=(),
            )
            continue

        if suffix == "gain_projector.2.weight":
            merged[key] = reduce_state_layout(
                old_param,
                new_param,
                prefix_shape=(3,),
                suffix_shape=(old_param.shape[1],),
            )
            continue

        if suffix == "gain_projector.2.bias":
            merged[key] = reduce_state_layout(
                old_param,
                new_param,
                prefix_shape=(3,),
                suffix_shape=(),
            )
            continue

        if suffix == "return_map.proj.weight":
            feature_channels = _return_map_feature_channels(old_param, old_state_shape)
            merged[key] = reduce_state_layout(
                old_param,
                new_param,
                prefix_shape=(old_param.shape[0], feature_channels),
                suffix_shape=(),
            )
            continue

        merged[key] = new_param
    return merged


def _mode_prune_copy_state_dict(
    old_sd: Dict[str, Tensor],
    new_sd: Dict[str, Tensor],
    *,
    old_state_shape: Tuple[int, ...],
    new_state_shape: Tuple[int, ...],
    pruned_axis: int,
    pruned_slice: int,
) -> Dict[str, Tensor]:
    def shrink_state_layout(
        old_param: Tensor,
        new_param: Tensor,
        *,
        prefix_shape: Tuple[int, ...],
        suffix_shape: Tuple[int, ...],
    ) -> Tensor:
        old_view = old_param.reshape(*prefix_shape, *old_state_shape, *suffix_shape)
        axis = len(prefix_shape) + pruned_axis
        keep_indices = [index for index in range(old_state_shape[pruned_axis]) if index != pruned_slice]
        index_tensor = torch.tensor(keep_indices, device=old_view.device)
        shrunk = old_view.index_select(axis, index_tensor)
        return shrunk.reshape_as(new_param).to(new_param.dtype)

    merged = {}
    for key, new_param in new_sd.items():
        block_match = _BLOCK_MIXER_KEY_RE.match(key)
        if key.startswith("cross_layer_proj.") and key.endswith(".weight"):
            if key not in old_sd:
                merged[key] = new_param
                continue

            old_param = old_sd[key]
            if old_param.shape == new_param.shape:
                merged[key] = old_param
                continue

            feature_channels = _cross_layer_feature_channels(old_param, old_state_shape)
            merged[key] = shrink_state_layout(
                old_param,
                new_param,
                prefix_shape=(old_param.shape[0], feature_channels),
                suffix_shape=(),
            )
            continue

        if block_match is None:
            merged[key] = old_sd.get(key, new_param)
            continue

        suffix = block_match.group(2)
        if key not in old_sd:
            merged[key] = new_param
            continue

        old_param = old_sd[key]
        if old_param.shape == new_param.shape:
            merged[key] = old_param
            continue

        if suffix in {"decay_logit", "input_logit", "recurrent_logit", "self_relation_logit"}:
            merged[key] = shrink_state_layout(old_param, new_param, prefix_shape=(), suffix_shape=())
            continue

        if suffix in {"signal_projector.magnitude_proj.bias", "signal_projector.phase_proj.bias"}:
            merged[key] = shrink_state_layout(old_param, new_param, prefix_shape=(), suffix_shape=())
            continue

        if suffix in {"signal_projector.magnitude_proj.weight", "signal_projector.phase_proj.weight"}:
            merged[key] = shrink_state_layout(
                old_param,
                new_param,
                prefix_shape=(),
                suffix_shape=(old_param.shape[1],),
            )
            continue

        if suffix == "gain_projector.0.weight":
            merged[key] = shrink_state_layout(
                old_param,
                new_param,
                prefix_shape=(old_param.shape[0], 2),
                suffix_shape=(),
            )
            continue

        if suffix == "gain_projector.2.weight":
            merged[key] = shrink_state_layout(
                old_param,
                new_param,
                prefix_shape=(3,),
                suffix_shape=(old_param.shape[1],),
            )
            continue

        if suffix == "gain_projector.2.bias":
            merged[key] = shrink_state_layout(
                old_param,
                new_param,
                prefix_shape=(3,),
                suffix_shape=(),
            )
            continue

        if suffix == "return_map.proj.weight":
            feature_channels = _return_map_feature_channels(old_param, old_state_shape)
            merged[key] = shrink_state_layout(
                old_param,
                new_param,
                prefix_shape=(old_param.shape[0], feature_channels),
                suffix_shape=(),
            )
            continue

        mode_weight_match = _MODE_WEIGHT_KEY_RE.match(suffix)
        if mode_weight_match is not None:
            mode_index = int(mode_weight_match.group(1))
            if mode_index != pruned_axis:
                merged[key] = old_param
                continue
            keep_indices = [index for index in range(old_state_shape[pruned_axis]) if index != pruned_slice]
            index_tensor = torch.tensor(keep_indices, device=old_param.device)
            merged[key] = old_param.index_select(0, index_tensor).index_select(1, index_tensor).to(new_param.dtype)
            continue

        merged[key] = new_param
    return merged


def _detach_state_element(s: object) -> object:
    if s is None:
        return None
    if isinstance(s, tuple):
        return tuple(_detach_state_element(x) for x in s)
    return s.detach()  # type: ignore[union-attr]


def _reset_state_element_for_streams(s: object, wrapped_mask: List[bool]) -> object:
    if s is None:
        return None
    if isinstance(s, tuple):
        return tuple(_reset_state_element_for_streams(x, wrapped_mask) for x in s)
    s = s.clone()  # type: ignore[union-attr]
    for i, wrapped in enumerate(wrapped_mask):
        if wrapped:
            s[i] = 0  # type: ignore[index]
    return s


def _reset_wrapped_stream_states(
    states: Tuple[Optional[Tensor], ...],
    wrapped_mask: List[bool],
) -> Tuple[Optional[Tensor], ...]:
    return tuple(_reset_state_element_for_streams(s, wrapped_mask) for s in states)  # type: ignore[return-value]


def _compute_batch_metrics(
    model: ReciprocatorLM,
    token_ids: Tensor,
    targets: Tensor,
    *,
    states: Optional[Sequence[Optional[Tensor]]] = None,
    chunk_size: Optional[int] = None,
    track_drift: bool = False,
) -> Tuple[Tensor, TrainingMetrics, Tuple[Optional[Tensor], ...], Optional[dict]]:
    drift_stats = None
    outputs = model(
        token_ids,
        states=states,
        chunk_size=chunk_size,
        track_drift=track_drift,
    )
    logits, next_states, *extra = outputs
    if track_drift and extra:
        drift_stats = extra[-1]
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions == targets).float().mean().item())
    return loss, _build_metrics(float(loss.item()), accuracy), next_states, drift_stats


def _record_residual_diagnostic_row(residual_diagnostics: list[dict], row: dict) -> None:
    step = row["step"]
    for existing in reversed(residual_diagnostics):
        if existing.get("step") == step:
            existing.update(row)
            return
    residual_diagnostics.append(row)


def _reciprocator_state_tensors(states: Sequence[object]) -> Tuple[Tensor, ...]:
    return tuple(state for state in states if isinstance(state, Tensor))


def _compute_validation_cross_memory_residual(
    model: ReciprocatorLM,
    token_ids: Tensor,
    *,
    states: Sequence[object],
) -> float:
    captured_kv: list[Tuple[Tensor, Tensor]] = []
    handles = []

    def capture_attention_cache(_module, _inputs, output) -> None:
        if not isinstance(output, tuple) or len(output) != 2:
            return
        kv_cache = output[1]
        if (
            isinstance(kv_cache, tuple)
            and len(kv_cache) == 2
            and isinstance(kv_cache[0], Tensor)
            and isinstance(kv_cache[1], Tensor)
        ):
            captured_kv.append((kv_cache[0].detach(), kv_cache[1].detach()))

    for block in model.blocks:
        if isinstance(block, LocalAttentionBlock):
            handles.append(block.register_forward_hook(capture_attention_cache))

    if not handles:
        return 0.0

    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(token_ids, states=states)
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)

    next_states = outputs[1]
    tensor_states = _reciprocator_state_tensors(next_states)
    if not captured_kv or not tensor_states:
        return 0.0

    residuals = [
        model._compute_cross_memory_residual(kv_cache, tensor_states)
        for kv_cache in captured_kv
    ]
    return float(sum(residuals) / len(residuals))


def _initial_validation_states(
    model: ReciprocatorLM,
    *,
    batch_size: int,
    device: torch.device,
    carry_states: Optional[Sequence[object]],
) -> Tuple[object, ...]:
    if carry_states is None:
        return model.initial_state(batch_size, device=device, dtype=torch.cfloat)
    return tuple(_detach_state_element(state) for state in carry_states)


def _format_growth_event_history(
    growth_event_history: Sequence[Tuple[int, Tuple[int, ...], Tuple[int, ...]]],
) -> list[tuple[int, list[int], list[int]]]:
    return [
        (int(event_step), list(old_shape), list(new_shape))
        for event_step, old_shape, new_shape in growth_event_history
    ]


def _build_validation_diagnostic_row(
    model: ReciprocatorLM,
    tokens: Tensor,
    *,
    step: int,
    val_metric: TrainingMetrics,
    config: TrainingConfig,
    carry_states: Optional[Sequence[object]],
    growth_event_history: Sequence[Tuple[int, Tuple[int, ...], Tuple[int, ...]]],
    device: torch.device,
) -> dict:
    with torch.random.fork_rng():
        sample_inputs, _ = sample_causal_lm_batch(
            tokens,
            config.batch_size,
            config.seq_len,
            device=device,
        )

    diagnostic_states = _initial_validation_states(
        model,
        batch_size=sample_inputs.shape[0],
        device=device,
        carry_states=carry_states,
    )
    cross_memory_residual = _compute_validation_cross_memory_residual(
        model,
        sample_inputs,
        states=diagnostic_states,
    )
    phase_stats = PhaseTrajectoryMonitor().record(model, sample_inputs)

    return {
        "step": step,
        "state_shape": list(config.state_shape),
        "growth_event_history": _format_growth_event_history(growth_event_history),
        "val_loss": val_metric.loss,
        "val_accuracy": val_metric.accuracy,
        "val_perplexity": val_metric.perplexity,
        "val_bpc": val_metric.bpc,
        "cross_memory_residual": cross_memory_residual,
        **asdict(phase_stats),
    }


def _build_model_from_config(
    config: TrainingConfig,
    dataset: TextDataset,
    device: torch.device,
) -> ReciprocatorLM:
    return ReciprocatorLM(
        vocab_size=dataset.vocab_size,
        hidden_size=config.hidden_size,
        state_shape=config.state_shape,
        num_layers=config.num_layers,
        ffn_expansion_factor=config.ffn_expansion_factor,
        readout_type=config.readout_type,
        token_magnitude_type=config.token_magnitude_type,
        phase_type=config.phase_type,
        token_phase=config.token_phase,
        enable_self_relation=config.enable_self_relation,
        dynamic_gains=config.dynamic_gains,
        gain_projector_rank=config.gain_projector_rank,
        enable_cross_layer_state=config.enable_cross_layer_state,
        coupling_type=config.coupling_type,
        low_frequency_gain=config.low_frequency_gain,
        low_frequency_sigma=config.low_frequency_sigma,
        high_frequency_gain=config.high_frequency_gain,
        high_frequency_cutoff=config.high_frequency_cutoff,
        dynamic_spectral_gains=config.dynamic_spectral_gains,
        anisotropic_spectral_gains=config.anisotropic_spectral_gains,
        wavelet_levels=config.wavelet_levels,
        normalization_type=config.normalization_type,
        token_frequencies=dataset.token_frequencies(),
        attention_every_k=config.attention_every_k,
        attention_num_heads=config.attention_num_heads,
        attention_window=config.attention_window,
        attention_position=config.attention_position,
        block_layout=config.block_layout,
    ).to(device)


def _try_dynamic_growth(
    model: ReciprocatorLM,
    config: TrainingConfig,
    dataset: TextDataset,
    device: torch.device,
    recent_losses: List[float],
    smoothed_mode_residual_norms: Tensor,
    reference_token_ids: Optional[Tensor] = None,
    effective_growth_residual_threshold: Optional[float] = None,
) -> Tuple[Optional[ReciprocatorLM], Optional[TrainingConfig]]:
    if not config.dynamic_mode_growth and not config.dynamic_rank_growth:
        return None, None

    growth_action = _select_dynamic_growth_action(
        config,
        smoothed_mode_residual_norms,
        recent_losses,
        effective_growth_residual_threshold=effective_growth_residual_threshold,
    )
    if growth_action is None:
        return None, None

    current_shape = config.state_shape
    growth_kind, grown_axis = growth_action
    if growth_kind == "rank":
        new_shape = tuple(current_shape) + (2,)
    elif growth_kind == "mode":
        new_shape = list(current_shape)
        new_shape[grown_axis] += 1
        new_shape = tuple(new_shape)
    else:
        raise ValueError(f"Unsupported growth kind: {growth_kind!r}")

    new_config = replace(config, state_shape=new_shape)
    new_model = _build_model_from_config(new_config, dataset, device)
    reference_contexts = None
    effective_mode_init = config.mode_init
    effective_rank_init = config.rank_init
    if (
        ((growth_kind == "mode" and effective_mode_init in {"orthogonal", "residual"})
        or (growth_kind == "rank" and effective_rank_init == "residual"))
        and reference_token_ids is None
    ):
        raise ValueError("reference_token_ids are required for residual/orthogonal growth initialization.")
    if (
        (growth_kind == "mode" and effective_mode_init in {"orthogonal", "residual"})
        or (growth_kind == "rank" and effective_rank_init == "residual")
    ):
        if growth_kind == "mode":
            reference_contexts = _collect_growth_reference_contexts(model, reference_token_ids)
        else:
            reference_contexts = _collect_rank_growth_reference_contexts(model, reference_token_ids)
    merged_sd = _pad_copy_state_dict(
        model.state_dict(),
        new_model.state_dict(),
        old_state_shape=tuple(current_shape),
        new_state_shape=tuple(new_shape),
        growth_kind=growth_kind,
        grown_axis=grown_axis,
        mode_init=effective_mode_init,
        rank_init=effective_rank_init,
        reference_contexts=reference_contexts,
    )
    new_model.load_state_dict(merged_sd)

    return new_model, new_config


def _pruned_max_state_shape(
    max_state_shape: Optional[Tuple[int, ...]],
    *,
    pruned_axis: int,
) -> Optional[Tuple[int, ...]]:
    if max_state_shape is None:
        return None
    if pruned_axis >= len(max_state_shape):
        return tuple(max_state_shape)
    return tuple(max_state_shape[:pruned_axis] + max_state_shape[pruned_axis + 1 :])


def _return_map_feature_channels(param: Tensor, state_shape: Tuple[int, ...]) -> int:
    state_size = math.prod(state_shape)
    if param.ndim != 2 or state_size <= 0 or param.shape[1] % state_size != 0:
        raise ValueError("return_map weights must be 2-D with an input size divisible by state_size.")
    return param.shape[1] // state_size


def _cross_layer_feature_channels(param: Tensor, state_shape: Tuple[int, ...]) -> int:
    state_size = math.prod(state_shape)
    if param.ndim != 2 or state_size <= 0 or param.shape[1] % state_size != 0:
        raise ValueError("cross_layer_proj weights must be 2-D with an input size divisible by state_size.")
    return param.shape[1] // state_size


def _try_dynamic_axis_pruning(
    model: ReciprocatorLM,
    config: TrainingConfig,
    dataset: TextDataset,
    device: torch.device,
    *,
    pruned_axis: int,
) -> Tuple[Optional[ReciprocatorLM], Optional[TrainingConfig]]:
    current_shape = tuple(config.state_shape)
    if len(current_shape) <= 1:
        return None, None

    new_shape = current_shape[:pruned_axis] + current_shape[pruned_axis + 1 :]
    new_config = replace(
        config,
        state_shape=new_shape,
        max_state_shape=_pruned_max_state_shape(config.max_state_shape, pruned_axis=pruned_axis),
    )
    new_model = _build_model_from_config(new_config, dataset, device)
    merged_sd = _prune_copy_state_dict(
        model.state_dict(),
        new_model.state_dict(),
        old_state_shape=current_shape,
        new_state_shape=new_shape,
        pruned_axis=pruned_axis,
    )
    new_model.load_state_dict(merged_sd)
    return new_model, new_config


def _try_dynamic_mode_size_pruning(
    model: ReciprocatorLM,
    config: TrainingConfig,
    dataset: TextDataset,
    device: torch.device,
    *,
    pruned_axis: int,
    pruned_slice: int,
) -> Tuple[Optional[ReciprocatorLM], Optional[TrainingConfig]]:
    current_shape = tuple(config.state_shape)
    if pruned_axis >= len(current_shape) or current_shape[pruned_axis] <= 1:
        return None, None

    new_shape = list(current_shape)
    new_shape[pruned_axis] -= 1
    new_shape = tuple(new_shape)
    new_config = replace(
        config,
        state_shape=new_shape,
        max_state_shape=_effective_max_state_shape(config),
    )
    new_model = _build_model_from_config(new_config, dataset, device)
    merged_sd = _mode_prune_copy_state_dict(
        model.state_dict(),
        new_model.state_dict(),
        old_state_shape=current_shape,
        new_state_shape=new_shape,
        pruned_axis=pruned_axis,
        pruned_slice=pruned_slice,
    )
    new_model.load_state_dict(merged_sd)
    return new_model, new_config


def _try_dynamic_mode_pruning(
    model: ReciprocatorLM,
    config: TrainingConfig,
    dataset: TextDataset,
    device: torch.device,
    smoothed_mode_pruning_residual_norms: Tensor,
    smoothed_mode_slice_activation_variances: Sequence[Tensor],
    prune_candidate_streaks: Sequence[int],
    mode_last_growth_steps: Sequence[int],
    axis_kinds: Sequence[str],
    *,
    step: int,
) -> Tuple[Optional[ReciprocatorLM], Optional[TrainingConfig]]:
    if not config.dynamic_mode_pruning:
        return None, None

    pruned_axis = _select_dynamic_mode_pruning_action(
        config,
        smoothed_mode_pruning_residual_norms,
        prune_candidate_streaks,
        mode_last_growth_steps,
        axis_kinds,
        step=step,
    )
    if pruned_axis is None:
        return None, None

    pruned_slice = _select_mode_slice_to_prune(
        config,
        smoothed_mode_slice_activation_variances,
        pruned_axis=pruned_axis,
    )

    return _try_dynamic_mode_size_pruning(
        model,
        config,
        dataset,
        device,
        pruned_axis=pruned_axis,
        pruned_slice=pruned_slice,
    )


def _try_dynamic_rank_pruning(
    model: ReciprocatorLM,
    config: TrainingConfig,
    dataset: TextDataset,
    device: torch.device,
    smoothed_mode_pruning_residual_norms: Tensor,
    prune_candidate_streaks: Sequence[int],
    mode_last_growth_steps: Sequence[int],
    axis_kinds: Sequence[str],
    *,
    step: int,
) -> Tuple[Optional[ReciprocatorLM], Optional[TrainingConfig]]:
    if not config.dynamic_rank_pruning:
        return None, None

    pruned_axis = _select_dynamic_rank_pruning_action(
        config,
        smoothed_mode_pruning_residual_norms,
        prune_candidate_streaks,
        mode_last_growth_steps,
        axis_kinds,
        step=step,
    )
    if pruned_axis is None:
        return None, None

    return _try_dynamic_axis_pruning(
        model,
        config,
        dataset,
        device,
        pruned_axis=pruned_axis,
    )


def _update_state_axis_kinds(
    previous_kinds: Sequence[str],
    previous_shape: Tuple[int, ...],
    current_shape: Tuple[int, ...],
) -> list[str]:
    shape_change = _detect_state_shape_change(previous_shape, current_shape)
    updated_kinds = list(previous_kinds)
    if shape_change is None:
        return updated_kinds

    change_kind, axis = shape_change
    if change_kind == "rank_growth":
        updated_kinds.append("rank")
        return updated_kinds
    if change_kind == "rank_prune":
        updated_kinds.pop(axis)
        return updated_kinds
    return updated_kinds


def _update_mode_last_growth_steps(
    previous_steps: Sequence[int],
    previous_shape: Tuple[int, ...],
    current_shape: Tuple[int, ...],
    *,
    step: int,
) -> list[int]:
    shape_change = _detect_state_shape_change(previous_shape, current_shape)
    updated_steps = list(previous_steps)
    if shape_change is None:
        return updated_steps

    change_kind, axis = shape_change
    if change_kind == "mode_growth":
        updated_steps[axis] = step
        return updated_steps
    if change_kind == "rank_growth":
        updated_steps.append(step)
        return updated_steps
    if change_kind == "rank_prune":
        updated_steps.pop(axis)
        return updated_steps
    return updated_steps


def evaluate_loss(
    model: ReciprocatorLM,
    tokens: Tensor,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    return evaluate_metrics(
        model,
        tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        eval_batches=eval_batches,
        device=device,
    ).loss


def evaluate_metrics(
    model: ReciprocatorLM,
    tokens: Tensor,
    *,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    if eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            inputs, targets = sample_causal_lm_batch(tokens, batch_size, seq_len, device=device)
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            predictions = logits.argmax(dim=-1)

            batch_token_count = int(targets.numel())
            total_loss += float(loss.item()) * batch_token_count
            total_correct += int((predictions == targets).sum().item())
            total_tokens += batch_token_count

    mean_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    return _build_metrics(mean_loss, accuracy)


def _normalize_training_config(config: TrainingConfig) -> TrainingConfig:
    normalized_lr_decay_style = config.lr_decay_style.strip().lower().replace("-", "_")
    normalized_generation_top_k = (
        None
        if config.generation_top_k is None or int(config.generation_top_k) <= 0
        else int(config.generation_top_k)
    )
    normalized_benchmark_prompt_lengths = tuple(int(length) for length in config.benchmark_prompt_lengths)
    if (
        normalized_lr_decay_style == config.lr_decay_style
        and normalized_generation_top_k == config.generation_top_k
        and normalized_benchmark_prompt_lengths == config.benchmark_prompt_lengths
    ):
        return config
    return replace(
        config,
        lr_decay_style=normalized_lr_decay_style,
        generation_top_k=normalized_generation_top_k,
        benchmark_prompt_lengths=normalized_benchmark_prompt_lengths,
    )


def _validate_training_config(config: TrainingConfig) -> None:
    canonicalize_coupling_type(config.coupling_type)
    if config.steps <= 0:
        raise ValueError("steps must be positive.")
    if config.eval_every <= 0:
        raise ValueError("eval_every must be positive.")
    if config.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative.")
    if config.lr_decay_style not in {"constant", "linear", "cosine"}:
        raise ValueError("lr_decay_style must be one of {'constant', 'linear', 'cosine'}.")
    if not 0.0 <= config.min_lr_scale <= 1.0:
        raise ValueError("min_lr_scale must be in the range [0.0, 1.0].")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive when provided.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if config.gain_projector_rank <= 0:
        raise ValueError("gain_projector_rank must be positive.")
    if config.growth_residual_threshold < 0:
        raise ValueError("growth_residual_threshold must be non-negative.")
    if config.post_growth_cooldown_checks < 0:
        raise ValueError("post_growth_cooldown_checks must be non-negative.")
    if config.post_growth_cooldown_threshold_scale < 1.0:
        raise ValueError("post_growth_cooldown_threshold_scale must be at least 1.0.")
    if config.residual_saturate_threshold < 0:
        raise ValueError("residual_saturate_threshold must be non-negative.")
    if not 0.0 <= config.growth_residual_ema_decay < 1.0:
        raise ValueError("growth_residual_ema_decay must be in the range [0.0, 1.0).")
    if config.chunk_size is not None and config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided.")
    if config.min_checks_before_first_growth < 0:
        raise ValueError("min_checks_before_first_growth must be non-negative.")
    if config.rank_growth_loss_ceiling < 0:
        raise ValueError("rank_growth_loss_ceiling must be non-negative.")
    if config.prune_threshold < 0:
        raise ValueError("prune_threshold must be non-negative.")
    if config.prune_sustain_steps < 0:
        raise ValueError("prune_sustain_steps must be non-negative.")
    if config.prune_min_steps < 0:
        raise ValueError("prune_min_steps must be non-negative.")
    if config.low_frequency_gain < 0.0:
        raise ValueError("low_frequency_gain must be non-negative.")
    if config.low_frequency_sigma <= 0.0:
        raise ValueError("low_frequency_sigma must be positive.")
    if config.high_frequency_gain < 0.0:
        raise ValueError("high_frequency_gain must be non-negative.")
    if not 0.0 <= config.high_frequency_cutoff <= 1.0:
        raise ValueError("high_frequency_cutoff must be in the range [0.0, 1.0].")
    if config.wavelet_levels is not None and config.wavelet_levels <= 0:
        raise ValueError("wavelet_levels must be positive when provided.")
    if config.generation_eval_samples < 0:
        raise ValueError("generation_eval_samples must be non-negative.")
    if config.generation_prompt_len <= 0:
        raise ValueError("generation_prompt_len must be positive.")
    if config.generation_new_tokens < 0:
        raise ValueError("generation_new_tokens must be non-negative.")
    if config.generation_temperature < 0.0:
        raise ValueError("generation_temperature must be non-negative.")
    if any(length <= 0 for length in config.benchmark_prompt_lengths):
        raise ValueError("benchmark_prompt_lengths must contain only positive integers.")
    if config.benchmark_prompt_lengths and config.benchmark_new_tokens <= 0:
        raise ValueError("benchmark_new_tokens must be positive when benchmarking is enabled.")
    if config.block_layout is not None:
        if not config.block_layout:
            raise ValueError("block_layout must contain at least one block.")
        if any(block not in {"attention", "reciprocator"} for block in config.block_layout):
            raise ValueError("block_layout entries must be 'attention' or 'reciprocator'.")
        if "attention" in config.block_layout and config.hidden_size % config.attention_num_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by attention_num_heads "
                f"({config.attention_num_heads}) when block_layout contains attention."
            )
        if "reciprocator" not in config.block_layout and (
            config.dynamic_mode_growth
            or config.dynamic_rank_growth
            or config.dynamic_mode_pruning
            or config.dynamic_rank_pruning
            or config.enable_cross_layer_state
        ):
            raise ValueError("reciprocator-only features require at least one reciprocator block.")
    if config.attention_every_k > 0 and config.attention_position not in {"before", "after"}:
        raise ValueError("attention_position must be 'before' or 'after'.")
    if config.attention_every_k > 0 and config.hidden_size % config.attention_num_heads != 0:
        raise ValueError(
            f"hidden_size ({config.hidden_size}) must be divisible by attention_num_heads "
            f"({config.attention_num_heads}) when attention_every_k > 0."
        )
    if config.attention_every_k < 0:
        raise ValueError("attention_every_k must be non-negative.")
    if config.attention_window <= 0:
        raise ValueError("attention_window must be positive.")


def _validate_training_dataset(config: TrainingConfig, dataset: TextDataset) -> None:
    if dataset.train_tokens.numel() <= config.seq_len:
        raise ValueError(
            f"Train split has {dataset.train_tokens.numel()} tokens, which is too small for seq_len={config.seq_len}."
        )
    if dataset.val_tokens.numel() <= config.seq_len:
        raise ValueError(
            f"Validation split has {dataset.val_tokens.numel()} tokens, "
            f"which is too small for seq_len={config.seq_len}."
        )
    if config.stateful_training:
        _min_stream_tokens = dataset.train_tokens.numel() // config.batch_size
        if _min_stream_tokens <= config.seq_len:
            raise ValueError(
                f"stateful_training requires at least seq_len tokens per stream, "
                f"but train corpus yields only {_min_stream_tokens} tokens per stream "
                f"(corpus={dataset.train_tokens.numel()}, batch_size={config.batch_size})."
            )


def _stateful_stream_starts_and_positions(
    *,
    total_train: int,
    batch_size: int,
    seq_len: int,
    completed_steps: int,
) -> Tuple[List[int], List[int]]:
    spacing = total_train // batch_size
    stream_starts = [i * spacing for i in range(batch_size)]
    if completed_steps <= 0:
        return stream_starts, list(stream_starts)

    stream_pos = []
    for start in stream_starts:
        steps_before_wrap = max((total_train - start - seq_len - 1) // seq_len + 1, 1)
        stream_pos.append(start + (completed_steps % steps_before_wrap) * seq_len)
    return stream_starts, stream_pos


def train_model(
    config: TrainingConfig,
    *,
    dataset: Optional[TextDataset] = None,
    step_callback: Optional[TrainingStepCallback] = None,
    resume_state: Optional[TrainingResumeState] = None,
) -> TrainingResult:
    config = _normalize_training_config(config)
    _validate_training_config(config)

    # Keep dataset-independent validation above dataset construction so
    # stateful stream-size checks cannot mask incompatible option errors.
    torch.manual_seed(config.seed)

    if dataset is None:
        dataset = build_corpus_dataset(
            config.corpus_name,
            max_chars=config.max_chars,
            val_fraction=config.val_fraction,
        )

    _validate_training_dataset(config, dataset)

    device = _resolve_device(config.device)
    _configure_tensor_dynamic_growth(device, enabled=config.tensor_dynamic_growth)
    model = _build_model_from_config(config, dataset, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    resume_step = 0
    if resume_state is not None:
        resume_step = int(resume_state.step)
        if resume_step < 0:
            raise ValueError("resume_state.step must be non-negative.")
        if resume_step > config.steps:
            raise ValueError("resume_state.step cannot exceed config.steps.")
        model.load_state_dict(resume_state.model_state_dict)
        if resume_state.optimizer_state_dict is not None:
            optimizer.load_state_dict(resume_state.optimizer_state_dict)
    else:
        _set_optimizer_learning_rate_scale(optimizer, _learning_rate_scale_for_step(config, 1))

    mode_residual_ema = None
    layer_mode_residual_ema = None
    mode_pruning_residual_ema = None
    mode_slice_activation_variance_ema = None
    prune_candidate_streaks = [0] * len(config.state_shape)
    mode_last_growth_steps = [0] * len(config.state_shape)
    growth_event_history: list[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = []
    state_axis_kinds = ["mode"] * len(config.state_shape)

    train_losses: list[float] = [] if resume_state is None else [float(loss) for loss in resume_state.train_losses]
    train_metrics: list[Tuple[int, TrainingMetrics]] = (
        [] if resume_state is None else [(int(step), metrics) for step, metrics in resume_state.train_metrics]
    )
    val_losses: list[Tuple[int, float]] = (
        [] if resume_state is None else [(int(step), float(loss)) for step, loss in resume_state.val_losses]
    )
    val_metrics: list[Tuple[int, TrainingMetrics]] = (
        [] if resume_state is None else [(int(step), metrics) for step, metrics in resume_state.val_metrics]
    )
    generation_samples: list[GenerationSample] = []
    runtime_benchmarks: list[RuntimeBenchmark] = []
    residual_diagnostics: list[dict] = []
    chunk_drift_history: list[dict] = []
    force_exact_steps = 0
    post_growth_cooldown_checks_remaining = 0

    # Stateful training: B independent corpus streams, state carried across chunks.
    carry_states: Optional[Tuple[Optional[Tensor], ...]] = None
    stream_pos: Optional[List[int]] = None
    stream_starts: Optional[List[int]] = None
    if config.stateful_training:
        total_train = dataset.train_tokens.numel()
        stream_starts, stream_pos = _stateful_stream_starts_and_positions(
            total_train=total_train,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            completed_steps=resume_step,
        )
        carry_states = tuple(None for _ in model.blocks)

    for step in range(resume_step + 1, config.steps + 1):
        _set_optimizer_learning_rate_scale(optimizer, _learning_rate_scale_for_step(config, step))
        if config.stateful_training:
            assert stream_pos is not None and stream_starts is not None
            inputs = torch.stack([
                dataset.train_tokens[p : p + config.seq_len]
                for p in stream_pos
            ]).to(device)
            targets = torch.stack([
                dataset.train_tokens[p + 1 : p + config.seq_len + 1]
                for p in stream_pos
            ]).to(device)
        else:
            inputs, targets = sample_causal_lm_batch(
                dataset.train_tokens,
                config.batch_size,
                config.seq_len,
                device=device,
            )
        growth_check_due = step > 0 and step % config.growth_check_interval == 0
        effective_growth_residual_threshold = _effective_growth_residual_threshold(
            config,
            post_growth_cooldown_checks_remaining,
        )
        if growth_check_due and any(hasattr(block, "mixer") for block in model.blocks):
            recent_chunk_drift_mean = None
            recent_chunk_drift_max = None
            if config.track_chunk_drift and len(chunk_drift_history) >= config.growth_check_interval:
                recent_chunk_window = chunk_drift_history[-config.growth_check_interval :]
                recent_chunk_drift_mean = (
                    sum(row["mean_drift"] for row in recent_chunk_window) / config.growth_check_interval
                )
                recent_chunk_drift_max = max(row["max_drift"] for row in recent_chunk_window)
                if recent_chunk_drift_mean > CHUNK_DRIFT_WARN_THRESHOLD:
                    warnings.warn(
                        f"Mean chunk drift {recent_chunk_drift_mean:.4f} exceeds "
                        f"{CHUNK_DRIFT_WARN_THRESHOLD:.2f} threshold. "
                        f"Consider reducing chunk_size (currently K={config.chunk_size}) "
                        "or switching to exact sequential (chunk_size=None).",
                    )

            current_mode_residual_norms = _compute_mode_residual_norms(model, inputs)
            mode_residual_ema = _update_ema(
                mode_residual_ema,
                current_mode_residual_norms,
                config.growth_residual_ema_decay,
            )
            current_layer_mode_residual_norms = None
            if config.record_residual_diagnostics and config.num_layers > 1:
                current_layer_mode_residual_norms = _compute_layer_mode_residual_norms(model, inputs)
                layer_mode_residual_ema = _update_ema(
                    layer_mode_residual_ema,
                    current_layer_mode_residual_norms,
                    config.growth_residual_ema_decay,
                )

            current_mode_pruning_residual_norms = None
            current_mode_slice_activation_variances = None
            if (
                config.dynamic_mode_pruning
                or config.dynamic_rank_pruning
                or config.record_residual_diagnostics
            ):
                current_mode_pruning_residual_norms = _compute_mode_pruning_residual_norms(model, inputs)
                mode_pruning_residual_ema = _update_ema(
                    mode_pruning_residual_ema,
                    current_mode_pruning_residual_norms,
                    config.growth_residual_ema_decay,
                )
                current_mode_slice_activation_variances = _compute_mode_slice_activation_variances(model, inputs)
                mode_slice_activation_variance_ema = _update_tensor_list_ema(
                    mode_slice_activation_variance_ema,
                    current_mode_slice_activation_variances,
                    config.growth_residual_ema_decay,
                )
                prune_candidate_streaks = _update_pruning_candidate_streaks(
                    prune_candidate_streaks,
                    mode_pruning_residual_ema,
                    threshold=config.prune_threshold,
                )

            if config.record_residual_diagnostics:
                _record_residual_diagnostic_row(
                    residual_diagnostics,
                    {
                        "step": step,
                        "state_shape": list(config.state_shape),
                        "axis_kinds": list(state_axis_kinds),
                        "mode_residual_norms": current_mode_residual_norms.tolist(),
                        "mode_residual_ema": mode_residual_ema.tolist(),
                        "effective_growth_residual_threshold": effective_growth_residual_threshold,
                        "post_growth_cooldown_checks_remaining": post_growth_cooldown_checks_remaining,
                        "layer_mode_residual_norms": (
                            []
                            if current_layer_mode_residual_norms is None
                            else current_layer_mode_residual_norms.tolist()
                        ),
                        "layer_mode_residual_ema": (
                            []
                            if layer_mode_residual_ema is None
                            else layer_mode_residual_ema.tolist()
                        ),
                        "mode_redundancy_norms": (
                            []
                            if current_mode_pruning_residual_norms is None
                            else current_mode_pruning_residual_norms.tolist()
                        ),
                        "mode_redundancy_ema": (
                            []
                            if mode_pruning_residual_ema is None
                            else mode_pruning_residual_ema.tolist()
                        ),
                        "mode_slice_activation_variance_ema": (
                            []
                            if mode_slice_activation_variance_ema is None
                            else [variances.tolist() for variances in mode_slice_activation_variance_ema]
                        ),
                        "prune_candidate_streaks": list(prune_candidate_streaks),
                        "recent_chunk_drift_mean": recent_chunk_drift_mean,
                        "recent_chunk_drift_max": recent_chunk_drift_max,
                    },
                )

        # Dynamic growth check (mode-size and/or rank), using EMA-smoothed residual norms.
        previous_shape = tuple(config.state_shape)
        shape_changed = False
        post_growth_cooldown_reset = False
        growth_check_ready = (
            growth_check_due and step // config.growth_check_interval >= config.min_checks_before_first_growth
        )
        if (
            (config.dynamic_mode_growth or config.dynamic_rank_growth)
            and mode_residual_ema is not None
            and growth_check_ready
        ):
            new_model, new_config = _try_dynamic_growth(
                model,
                config,
                dataset,
                device,
                train_losses,
                mode_residual_ema,
                reference_token_ids=inputs,
                effective_growth_residual_threshold=effective_growth_residual_threshold,
            )
            if new_model is not None:
                model = new_model
                config = new_config
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                _set_optimizer_learning_rate_scale(optimizer, _learning_rate_scale_for_step(config, step))
                mode_residual_ema = None
                mode_pruning_residual_ema = None
                mode_slice_activation_variance_ema = None
                prune_candidate_streaks = [0] * len(config.state_shape)
                mode_last_growth_steps = _update_mode_last_growth_steps(
                    mode_last_growth_steps,
                    previous_shape,
                    tuple(config.state_shape),
                    step=step,
                )
                state_axis_kinds = _update_state_axis_kinds(
                    state_axis_kinds,
                    previous_shape,
                    tuple(config.state_shape),
                )
                new_shape = tuple(config.state_shape)
                if new_shape != previous_shape:
                    growth_event_history.append((step, previous_shape, new_shape))
                    post_growth_cooldown_checks_remaining = config.post_growth_cooldown_checks
                    post_growth_cooldown_reset = True
                if config.stateful_training:
                    carry_states = tuple(None for _ in model.blocks)
                force_exact_steps = config.growth_check_interval
                shape_changed = True

        if (
            mode_pruning_residual_ema is not None
            and (config.dynamic_mode_pruning or config.dynamic_rank_pruning)
            and growth_check_due
            and not shape_changed
        ):
            rank_pruned_model, rank_pruned_config = _try_dynamic_rank_pruning(
                model,
                config,
                dataset,
                device,
                mode_pruning_residual_ema,
                prune_candidate_streaks,
                mode_last_growth_steps,
                state_axis_kinds,
                step=step,
            )
            if rank_pruned_model is not None:
                new_model, new_config = rank_pruned_model, rank_pruned_config
            else:
                new_model, new_config = _try_dynamic_mode_pruning(
                    model,
                    config,
                    dataset,
                    device,
                    mode_pruning_residual_ema,
                    mode_slice_activation_variance_ema if mode_slice_activation_variance_ema is not None else [],
                    prune_candidate_streaks,
                    mode_last_growth_steps,
                    state_axis_kinds,
                    step=step,
                )
            if new_model is not None:
                model = new_model
                config = new_config
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                _set_optimizer_learning_rate_scale(optimizer, _learning_rate_scale_for_step(config, step))
                mode_residual_ema = None
                mode_pruning_residual_ema = None
                mode_slice_activation_variance_ema = None
                prune_candidate_streaks = [0] * len(config.state_shape)
                mode_last_growth_steps = _update_mode_last_growth_steps(
                    mode_last_growth_steps,
                    previous_shape,
                    tuple(config.state_shape),
                    step=step,
                )
                state_axis_kinds = _update_state_axis_kinds(
                    state_axis_kinds,
                    previous_shape,
                    tuple(config.state_shape),
                )
                if config.stateful_training:
                    carry_states = tuple(None for _ in model.blocks)
                force_exact_steps = config.growth_check_interval

        if (
            growth_check_due
            and post_growth_cooldown_checks_remaining > 0
            and not post_growth_cooldown_reset
        ):
            post_growth_cooldown_checks_remaining -= 1

        model.train()
        optimizer.zero_grad()
        effective_chunk_size = None if force_exact_steps > 0 else config.chunk_size
        if force_exact_steps > 0:
            force_exact_steps -= 1
        loss, batch_metrics, next_states, drift_stats = _compute_batch_metrics(
            model,
            inputs,
            targets,
            states=carry_states,
            chunk_size=effective_chunk_size,
            track_drift=config.track_chunk_drift,
        )
        if config.track_chunk_drift and drift_stats is not None:
            chunk_drift_history.append({"step": step, **drift_stats})
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        train_losses.append(batch_metrics.loss)
        train_metrics.append((step, batch_metrics))

        if config.stateful_training:
            assert stream_pos is not None and stream_starts is not None
            total_train = dataset.train_tokens.numel()
            stream_pos = [p + config.seq_len for p in stream_pos]
            wrapped_mask = [p + config.seq_len + 1 > total_train for p in stream_pos]
            stream_pos = [stream_starts[i] if wrapped_mask[i] else stream_pos[i] for i in range(config.batch_size)]
            carry_states = tuple(_detach_state_element(s) for s in next_states)
            if any(wrapped_mask):
                carry_states = _reset_wrapped_stream_states(carry_states, wrapped_mask)

        should_evaluate = step == 1 or step == config.steps or step % config.eval_every == 0
        if should_evaluate:
            val_metric = evaluate_metrics(
                model,
                dataset.val_tokens,
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                eval_batches=config.eval_batches,
                device=device,
            )
            val_losses.append((step, val_metric.loss))
            val_metrics.append((step, val_metric))
            _record_residual_diagnostic_row(
                residual_diagnostics,
                _build_validation_diagnostic_row(
                    model,
                    dataset.val_tokens,
                    step=step,
                    val_metric=val_metric,
                    config=config,
                    carry_states=carry_states,
                    growth_event_history=growth_event_history,
                    device=device,
                ),
            )

        if step_callback is not None:
            step_callback(
                step,
                model,
                optimizer,
                dataset,
                config,
                train_losses,
                val_losses,
                train_metrics,
                val_metrics,
                residual_diagnostics,
                device,
            )

    generation_samples = evaluate_generation_samples(
        model,
        dataset,
        prompt_len=config.generation_prompt_len,
        max_new_tokens=config.generation_new_tokens,
        num_samples=config.generation_eval_samples,
        temperature=config.generation_temperature,
        top_k=config.generation_top_k,
        device=device,
    )
    runtime_benchmarks = benchmark_streaming_inference(
        model,
        dataset,
        prompt_lengths=config.benchmark_prompt_lengths,
        decode_tokens=config.benchmark_new_tokens,
        device=device,
    )

    return TrainingResult(
        config=config,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        train_losses=train_losses,
        train_metrics=train_metrics,
        val_losses=val_losses,
        val_metrics=val_metrics,
        generation_samples=generation_samples,
        runtime_benchmarks=runtime_benchmarks,
        residual_diagnostics=residual_diagnostics,
        chunk_drift_history=chunk_drift_history,
        device=device,
    )


__all__ = [
    "CharTokenizer",
    "GenerationSample",
    "LispGRPOConfig",
    "RLStepMetrics",
    "RLTrainingResult",
    "RuntimeBenchmark",
    "SampleRecord",
    "TextDataset",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingResumeState",
    "TrainingResult",
    "TrainingStepCallback",
    "benchmark_streaming_inference",
    "build_corpus_dataset",
    "build_lisp_tokenizer",
    "build_text_dataset",
    "evaluate_generation_samples",
    "evaluate_loss",
    "evaluate_metrics",
    "sample_causal_lm_batch",
    "train_lisp_grpo",
    "train_model",
]
