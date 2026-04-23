from reciprocator.rl.problem_gen import DifficultyConfig, ProblemExample
from reciprocator.rl.reward import RewardFunction
from reciprocator.rl.lisp_eval import parse_program


def test_stage_one_reward_scores_value_outputs() -> None:
    problem = ProblemExample(
        prompt_expression="(+ 1 2)",
        expected_result=3,
        difficulty=DifficultyConfig(stage=1, depth=1, operator_set=("+",), value_range=(0, 4)),
    )
    reward = RewardFunction()

    assert reward.score_output(problem, "3").reward == 1.0
    assert reward.score_output(problem, "4").reward == 0.5
    assert reward.score_output(problem, "(+").reward == 0.0


def test_stage_three_reward_scores_quoted_structure_shape() -> None:
    expected = RewardFunction()
    problem = ProblemExample(
        prompt_expression="(quote (+ 1 2))",
        expected_result=parse_program("(+ 1 2)")[0],
        difficulty=DifficultyConfig(stage=3),
    )

    assert expected.score_output(problem, "(+ 1 2)").reward == 1.0
    assert expected.score_output(problem, "(* 9 9)").reward == 0.5
    assert expected.score_output(problem, "3").reward == 0.0


def test_stage_five_reward_accepts_symbolic_equivalence() -> None:
    from reciprocator.rl.lisp_eval import parse_program

    problem = ProblemExample(
        prompt_expression="(simplify (quote (+ x 0)))",
        expected_result=parse_program("x")[0],
        difficulty=DifficultyConfig(stage=5),
    )
    reward = RewardFunction()

    assert reward.score_output(problem, "x").reward == 1.0
    assert reward.score_output(problem, "(+ x 0)").reward == 1.0
    assert reward.score_output(problem, "(+ x 1)").reward == 0.2


def test_stage_six_reward_scales_with_valid_proof_steps() -> None:
    from reciprocator.rl.lisp_eval import parse_program

    premise = parse_program("(+ x 0)")[0]
    target = parse_program("x")[0]
    problem = ProblemExample(
        prompt_expression="(prove x from (+ x 0))",
        expected_result=target,
        difficulty=DifficultyConfig(stage=6),
        metadata={"premises": (premise,), "target": target},
    )
    reward = RewardFunction()

    complete = reward.score_output(problem, "(step simplify-add-zero (+ x 0) x)")
    invalid = reward.score_output(problem, "(step commute-mul (+ x 0) x)")

    assert complete.reward == 1.0
    assert complete.error_type == "correct"
    assert invalid.reward < 1.0
    assert invalid.error_type == "invalid_proof_step"
