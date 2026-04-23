from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import random

from .problem_gen import DifficultyConfig, default_difficulty_for_stage


@dataclass(frozen=True)
class CurriculumSnapshot:
    current_stage: int
    stage_distribution: dict[int, float]
    success_rates: dict[int, float]


class CurriculumController:
    def __init__(
        self,
        *,
        stage_difficulties: dict[int, DifficultyConfig] | None = None,
        current_stage: int = 1,
        promotion_threshold: float = 0.8,
        demotion_floor: float = 0.3,
        history_window: int = 20,
        harder_stage_mix: float = 0.3,
    ) -> None:
        self.stage_difficulties = stage_difficulties or {stage: default_difficulty_for_stage(stage) for stage in range(1, 8)}
        self.current_stage = current_stage
        self.promotion_threshold = promotion_threshold
        self.demotion_floor = demotion_floor
        self.history_window = history_window
        self.harder_stage_mix = harder_stage_mix
        self._history = defaultdict(lambda: deque(maxlen=history_window))

    @property
    def max_stage(self) -> int:
        return max(self.stage_difficulties)

    def stage_distribution(self) -> dict[int, float]:
        distribution = {self.current_stage: 1.0}
        if self.current_stage < self.max_stage:
            harder_weight = self.harder_stage_mix
            distribution = {
                self.current_stage: 1.0 - harder_weight,
                self.current_stage + 1: harder_weight,
            }
        return distribution

    def sample_difficulties(self, batch_size: int, rng: random.Random) -> list[DifficultyConfig]:
        distribution = self.stage_distribution()
        stages = list(distribution)
        weights = [distribution[stage] for stage in stages]
        return [self.stage_difficulties[rng.choices(stages, weights=weights, k=1)[0]] for _ in range(batch_size)]

    def record_batch(self, stage_rewards: list[tuple[int, float]]) -> CurriculumSnapshot:
        for stage, reward in stage_rewards:
            self._history[stage].append(1.0 if reward >= 1.0 else 0.0)

        current_success = self.success_rate(self.current_stage)
        if len(self._history[self.current_stage]) >= self.history_window:
            if current_success >= self.promotion_threshold and self.current_stage < self.max_stage:
                self.current_stage += 1
            elif current_success <= self.demotion_floor and self.current_stage > 1:
                self.current_stage -= 1

        return self.snapshot()

    def success_rate(self, stage: int) -> float:
        history = self._history[stage]
        return 0.0 if not history else sum(history) / len(history)

    def snapshot(self) -> CurriculumSnapshot:
        return CurriculumSnapshot(
            current_stage=self.current_stage,
            stage_distribution=self.stage_distribution(),
            success_rates={stage: self.success_rate(stage) for stage in self.stage_difficulties},
        )


__all__ = [
    "CurriculumController",
    "CurriculumSnapshot",
]
