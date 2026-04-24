from __future__ import annotations

import copy
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F

from .curriculum import CurriculumController
from .phase_monitor import PhaseTrajectoryMonitor, PhaseTrajectoryStats
from .problem_gen import DifficultyConfig, ProblemExample, ProblemGenerator
from .reward import RewardFunction, RewardResult
from ..model import ReciprocatorLM

if TYPE_CHECKING:
    from ..training import CharTokenizer

DEFAULT_ALPHABET = " \n()[]{}:+-*/<>=abcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class LispGRPOConfig:
    steps: int = 20
    batch_size: int = 8
    group_size: int = 4
    max_completion_tokens: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    temperature: float = 1.0
    top_k: Optional[int] = None
    kl_beta: float = 0.02
    device: str = "cpu"
    seed: int = 0
    hidden_size: int = 32
    state_shape: tuple[int, ...] = (2, 2)
    num_layers: int = 1
    ffn_expansion_factor: int = 2
    readout_type: str = "phase_aware"
    token_magnitude_type: str = "inverse_frequency_learned"
    phase_type: str = "rope"
    token_phase: str = "semantic"
    enable_self_relation: bool = False
    coupling_type: str = "sequential"
    normalization_type: str = "frobenius"
    sample_record_limit: int = 4
    stage1_wrong_reward: float = 0.5


@dataclass(frozen=True)
class SampleRecord:
    stage: int
    prompt_expression: str
    completion: str
    expected_result: str
    reward: float
    error_type: str
    advantage: float
    mean_phase_variance: float
    phase_delta_variance: float


@dataclass(frozen=True)
class RLStepMetrics:
    step: int
    current_stage: int
    stage_distribution: dict[int, float]
    mean_reward: float
    success_rate: float
    mean_kl: float
    grad_norm: float
    mean_phase_variance: float
    mean_phase_delta: float
    loss: float
    error_counts: dict[str, int]
    samples: tuple[SampleRecord, ...]


@dataclass
class RLTrainingResult:
    config: LispGRPOConfig
    tokenizer: "CharTokenizer"
    model: ReciprocatorLM
    curriculum: CurriculumController
    step_metrics: list[RLStepMetrics]
    device: torch.device


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _sample_next_token(logits: torch.Tensor, *, temperature: float, top_k: Optional[int]) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits / temperature
    if top_k is not None and top_k > 0 and top_k < scaled.shape[-1]:
        top_values, top_indices = torch.topk(scaled, k=top_k, dim=-1)
        probabilities = torch.softmax(top_values, dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1)
        return top_indices.gather(-1, sampled)
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def build_lisp_tokenizer(extra_characters: str = "") -> "CharTokenizer":
    from ..training import CharTokenizer

    return CharTokenizer.from_text(DEFAULT_ALPHABET + extra_characters)


def build_shared_tokenizer(*corpus_names: str) -> "CharTokenizer":
    """Build a tokenizer covering one or more training corpora plus the Lisp character set.

    Produces a single vocabulary that can be used for both supervised LM training
    and RL mathematical reasoning, so model checkpoints transfer exactly.
    With no corpus names, builds from the Lisp alphabet alone.
    """
    from ..corpora import read_corpus_text
    from ..training import CharTokenizer

    parts = [DEFAULT_ALPHABET]
    for name in corpus_names:
        parts.append(read_corpus_text(name))
    return CharTokenizer.from_text("".join(parts))


def _build_model(config: LispGRPOConfig, tokenizer: "CharTokenizer", device: torch.device) -> ReciprocatorLM:
    token_frequencies = torch.ones(tokenizer.vocab_size, dtype=torch.float32, device=device)
    return ReciprocatorLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        state_shape=config.state_shape,
        num_layers=config.num_layers,
        ffn_expansion_factor=config.ffn_expansion_factor,
        readout_type=config.readout_type,
        token_magnitude_type=config.token_magnitude_type,
        phase_type=config.phase_type,
        token_phase=config.token_phase,
        enable_self_relation=config.enable_self_relation,
        coupling_type=config.coupling_type,
        normalization_type=config.normalization_type,
        token_frequencies=token_frequencies,
    ).to(device)


def _validate_config(config: LispGRPOConfig) -> None:
    if config.steps <= 0 or config.batch_size <= 0 or config.group_size <= 1:
        raise ValueError("steps and batch_size must be positive; group_size must be greater than 1.")
    if config.max_completion_tokens <= 0 or config.learning_rate <= 0.0:
        raise ValueError("max_completion_tokens and learning_rate must be positive.")
    if config.weight_decay < 0.0 or config.kl_beta < 0.0 or config.temperature < 0.0:
        raise ValueError("weight_decay, kl_beta, and temperature must be non-negative.")
    if not 0.0 <= config.stage1_wrong_reward <= 1.0:
        raise ValueError("stage1_wrong_reward must be in [0, 1].")


def _sample_completion(
    model: ReciprocatorLM,
    tokenizer: "CharTokenizer",
    prompt: str,
    *,
    max_completion_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: torch.device,
) -> tuple[str, tuple[int, ...]]:
    prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
    token_ids: list[int] = []
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, states = model(prompt_tokens, position_offset=0)
            current_logits = logits[:, -1]
            position = int(prompt_tokens.shape[1])
            for _ in range(max_completion_tokens):
                next_token = _sample_next_token(current_logits, temperature=temperature, top_k=top_k)
                token_id = int(next_token.item())
                token_ids.append(token_id)
                if tokenizer.decode([token_id]) == "\n":
                    break
                if len(token_ids) == max_completion_tokens:
                    break
                logits, states = model(next_token, states=states, position_offset=position)
                current_logits = logits[:, -1]
                position += 1
    finally:
        model.train(training)
    return tokenizer.decode(token_ids).rstrip("\n"), tuple(token_ids)


def _sequence_statistics(
    model: ReciprocatorLM,
    reference_model: ReciprocatorLM,
    tokenizer: "CharTokenizer",
    prompt: str,
    completion_ids: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
    completion = torch.tensor(completion_ids, dtype=torch.long, device=device)
    if completion.numel() == 0:
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        return zero, zero

    if completion.numel() > 1:
        input_tokens = torch.cat([prompt_tokens.squeeze(0), completion[:-1]], dim=0).unsqueeze(0)
    else:
        input_tokens = prompt_tokens

    logits, _ = model(input_tokens, position_offset=0)
    with torch.no_grad():
        ref_logits, _ = reference_model(input_tokens, position_offset=0)

    output_start = int(prompt_tokens.shape[1] - 1)
    output_stop = output_start + int(completion.numel())
    selected_logits = logits[:, output_start:output_stop, :]
    selected_ref_logits = ref_logits[:, output_start:output_stop, :]
    log_probs = F.log_softmax(selected_logits, dim=-1)
    with torch.no_grad():
        ref_log_probs = F.log_softmax(selected_ref_logits, dim=-1)

    token_log_probs = log_probs.gather(-1, completion.view(1, -1, 1)).squeeze(-1)
    probabilities = log_probs.exp()
    kl_terms = torch.sum(probabilities * (log_probs - ref_log_probs), dim=-1)
    return token_log_probs.mean(), kl_terms.mean()


def _group_relative_advantages(rewards: list[float]) -> list[float]:
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    if variance <= 1e-8:
        return [0.0 for _ in rewards]
    std = variance ** 0.5
    return [(reward - mean_reward) / std for reward in rewards]


def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to("cpu")
        total += float(grad.abs().pow(2).sum().item())
    return total**0.5


def _clip_gradient_norm(parameters, max_norm: float) -> float:
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    total_norm = _gradient_norm(parameters)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return total_norm


def _record_samples(
    problems: list[ProblemExample],
    outputs: list[list[str]],
    rewards: list[list[RewardResult]],
    advantages: list[list[float]],
    phase_stats: list[list[PhaseTrajectoryStats]],
    *,
    limit: int,
) -> tuple[SampleRecord, ...]:
    samples: list[SampleRecord] = []
    for problem, problem_outputs, problem_rewards, problem_advantages, problem_phase_stats in zip(
        problems, outputs, rewards, advantages, phase_stats
    ):
        for output, reward, advantage, stats in zip(problem_outputs, problem_rewards, problem_advantages, problem_phase_stats):
            samples.append(
                SampleRecord(
                    stage=problem.difficulty.stage,
                    prompt_expression=problem.prompt_expression,
                    completion=output,
                    expected_result=problem.expected_result_text,
                    reward=reward.reward,
                    error_type=reward.error_type,
                    advantage=advantage,
                    mean_phase_variance=stats.mean_phase_variance,
                    phase_delta_variance=stats.phase_delta_variance,
                )
            )
            if len(samples) >= limit:
                return tuple(samples)
    return tuple(samples)


def train_lisp_grpo(
    config: LispGRPOConfig,
    *,
    tokenizer: Optional["CharTokenizer"] = None,
    model: Optional[ReciprocatorLM] = None,
    curriculum: Optional[CurriculumController] = None,
    generator: Optional[ProblemGenerator] = None,
    reward_function: Optional[RewardFunction] = None,
    phase_monitor: Optional[PhaseTrajectoryMonitor] = None,
    step_callback: Optional[Callable[[RLStepMetrics, ReciprocatorLM, CurriculumController], None]] = None,
) -> RLTrainingResult:
    _validate_config(config)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = _resolve_device(config.device)
    tokenizer = tokenizer or build_lisp_tokenizer()
    model = (model or _build_model(config, tokenizer, device)).to(device)
    reference_model = copy.deepcopy(model).to(device).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    curriculum = curriculum or CurriculumController()
    generator = generator or ProblemGenerator(rng)
    reward_function = reward_function or RewardFunction(stage_one_wrong_reward=config.stage1_wrong_reward)
    phase_monitor = phase_monitor or PhaseTrajectoryMonitor()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    step_metrics: list[RLStepMetrics] = []
    model.train()

    for step in range(1, config.steps + 1):
        difficulties = curriculum.sample_difficulties(config.batch_size, rng)
        problems = generator.generate_batch(difficulties)

        losses: list[torch.Tensor] = []
        kl_values: list[float] = []
        outputs: list[list[str]] = []
        rewards: list[list[RewardResult]] = []
        advantages: list[list[float]] = []
        phase_rows: list[list[PhaseTrajectoryStats]] = []
        stage_rewards: list[tuple[int, float]] = []
        error_counts: dict[str, int] = {}

        for problem in problems:
            problem_outputs: list[str] = []
            completion_ids: list[tuple[int, ...]] = []
            problem_phase_rows: list[PhaseTrajectoryStats] = []
            for _ in range(config.group_size):
                prompt_text = problem.prompt_expression + "\n"
                completion, sampled_ids = _sample_completion(
                    model,
                    tokenizer,
                    prompt_text,
                    max_completion_tokens=config.max_completion_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    device=device,
                )
                problem_outputs.append(completion)
                completion_ids.append(sampled_ids)
                prompt_ids = tokenizer.encode(prompt_text).to(device)
                full_ids = torch.cat(
                    [
                        prompt_ids,
                        torch.tensor(sampled_ids, dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                problem_phase_rows.append(
                    phase_monitor.record(
                        model,
                        full_ids,
                        output_start=int(prompt_ids.numel()),
                    )
                )

            reward_rows = reward_function.score_outputs(problem, problem_outputs)
            reward_values = [row.reward for row in reward_rows]
            problem_advantages = _group_relative_advantages(reward_values)

            for reward_row in reward_rows:
                stage_rewards.append((problem.difficulty.stage, reward_row.reward))
                error_counts[reward_row.error_type] = error_counts.get(reward_row.error_type, 0) + 1

            for advantage, sampled_ids in zip(problem_advantages, completion_ids):
                log_prob, kl = _sequence_statistics(
                    model,
                    reference_model,
                    tokenizer,
                    problem.prompt_expression + "\n",
                    sampled_ids,
                    device=device,
                )
                losses.append((-advantage * log_prob) + config.kl_beta * kl)
                kl_values.append(float(kl.detach().item()))

            outputs.append(problem_outputs)
            rewards.append(reward_rows)
            advantages.append(problem_advantages)
            phase_rows.append(problem_phase_rows)

        optimizer.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = _gradient_norm(model.parameters())
        if config.grad_clip_norm is not None:
            grad_norm = _clip_gradient_norm(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        snapshot = curriculum.record_batch(stage_rewards)
        flat_rewards = [reward.reward for group in rewards for reward in group]
        flat_phase = [stats for group in phase_rows for stats in group]
        success_rate = sum(1.0 for reward in flat_rewards if reward >= 1.0) / len(flat_rewards)

        metrics = RLStepMetrics(
            step=step,
            current_stage=snapshot.current_stage,
            stage_distribution=snapshot.stage_distribution,
            mean_reward=sum(flat_rewards) / len(flat_rewards),
            success_rate=success_rate,
            mean_kl=sum(kl_values) / len(kl_values),
            grad_norm=grad_norm,
            mean_phase_variance=sum(stats.mean_phase_variance for stats in flat_phase) / len(flat_phase),
            mean_phase_delta=sum(stats.mean_phase_delta for stats in flat_phase) / len(flat_phase),
            loss=float(loss.detach().item()),
            error_counts=error_counts,
            samples=_record_samples(
                problems,
                outputs,
                rewards,
                advantages,
                phase_rows,
                limit=config.sample_record_limit,
            ),
        )
        step_metrics.append(metrics)
        if step_callback is not None:
            step_callback(metrics, model, curriculum)

    return RLTrainingResult(
        config=config,
        tokenizer=tokenizer,
        model=model,
        curriculum=curriculum,
        step_metrics=step_metrics,
        device=device,
    )


__all__ = [
    "LispGRPOConfig",
    "RLStepMetrics",
    "RLTrainingResult",
    "SampleRecord",
    "build_lisp_tokenizer",
    "build_shared_tokenizer",
    "train_lisp_grpo",
]
