import random

from reciprocator.rl.curriculum import CurriculumController


def test_curriculum_mixes_current_and_next_stage() -> None:
    curriculum = CurriculumController(current_stage=2, harder_stage_mix=0.3)

    assert curriculum.stage_distribution() == {2: 0.7, 3: 0.3}


def test_curriculum_promotes_and_demotes_by_recent_success_rate() -> None:
    curriculum = CurriculumController(current_stage=1, history_window=2, promotion_threshold=0.8, demotion_floor=0.3)

    snapshot = curriculum.record_batch([(1, 1.0), (1, 1.0)])
    assert snapshot.current_stage == 2

    snapshot = curriculum.record_batch([(2, 0.0), (2, 0.0)])
    assert snapshot.current_stage == 1


def test_curriculum_samples_difficulty_without_skipping_stages() -> None:
    curriculum = CurriculumController(current_stage=3)
    difficulties = curriculum.sample_difficulties(20, random.Random(0))
    stages = {difficulty.stage for difficulty in difficulties}

    assert stages <= {3, 4}
