from reciprocator.rl.problem_gen import DifficultyConfig, ProblemExample
from reciprocator.rl.reward import RewardFunction


def test_stage_one_wrong_reward_can_be_sharpened() -> None:
    problem = ProblemExample(
        prompt_expression="(+ 1 2)",
        expected_result=3,
        difficulty=DifficultyConfig(stage=1, depth=1),
    )

    assert RewardFunction().score_output(problem, "4").reward == 0.5
    sharpened = RewardFunction(stage_one_wrong_reward=0.1)

    result = sharpened.score_output(problem, "4")

    assert result.reward == 0.1
    assert result.error_type == "wrong_result"
