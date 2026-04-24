import random

import pytest

from reciprocator.rl.lisp_eval import (
    EvalError,
    LispEvaluator,
    ParseError,
    evaluate_program,
    parse_program,
    render_value,
    validate_proof,
    values_equal,
)
from reciprocator.rl.problem_gen import (
    DifficultyConfig,
    ProblemExample,
    ProblemGenerator,
    default_difficulty_for_stage,
)
from reciprocator.rl.reward import RewardFunction


@pytest.mark.parametrize(
    ("stage", "source", "expected"),
    [
        (1, "(+ 1 (* 2 3))", "7"),
        (1, "(/ 8 2)", "4"),
        (2, "(let [x (+ 3 4)] (* x 2))", "14"),
        (2, "(and (> 3 2) (not (= 1 2)))", ":true"),
        (3, "(quote (+ 1 2))", "(+ 1 2)"),
        (3, "(list :eval (eval (quote (+ 1 2))))", "(:eval 3)"),
        (4, "(if (> 3 0) 4 0)", "4"),
        (4, "(cond (= 0 1) 0 :else 9)", "9"),
        (4, "((lambda [x] (+ x 2)) 5)", "7"),
        (5, "(simplify (quote (+ x 0)))", "x"),
        (5, "(substitute (quote (+ x 1)) (quote x) 4)", "5"),
        (7, "(get [1 2 3] 1)", "2"),
        (7, "(->> [1 2 3] (mapv (partial + 1)) (reduce + 0))", "9"),
    ],
)
def test_phase_r0_evaluator_forms_by_stage(stage: int, source: str, expected: str) -> None:
    assert render_value(evaluate_program(source, stage=stage)) == expected


def test_phase_r0_proof_validator_accepts_valid_stage_six_proof() -> None:
    target = parse_program("x")[0]
    premise = parse_program("(+ x 0)")[0]

    proof = validate_proof(
        "(step simplify-add-zero (+ x 0) x)",
        target=target,
        premises=(premise,),
    )

    assert proof.complete is True
    assert proof.valid_fraction == 1.0


def test_phase_r0_evaluator_reports_parse_eval_and_stage_gate_errors() -> None:
    with pytest.raises(ParseError):
        parse_program("(+")

    with pytest.raises(EvalError):
        evaluate_program("(/ 1 0)", stage=1)

    gated_forms = [
        (1, "(let [x 1] x)"),
        (2, "(quote x)"),
        (3, "(if :true 1 0)"),
        (4, "(simplify (quote (+ x 0)))"),
        (6, "[1 2 3]"),
    ]
    for stage, source in gated_forms:
        with pytest.raises(EvalError):
            evaluate_program(source, stage=stage)


def test_phase_r0_problem_generator_matches_fresh_evaluator_for_1000_examples_per_stage() -> None:
    for stage in range(1, 8):
        generator = ProblemGenerator(random.Random(stage))
        difficulty = default_difficulty_for_stage(stage)

        for _ in range(1000):
            problem = generator.generate_problem(difficulty)
            parse_program(problem.prompt_expression)
            if stage == 6:
                proof = validate_proof(
                    problem.metadata["example_solution"],
                    target=problem.metadata["target"],
                    premises=tuple(problem.metadata["premises"]),
                )
                assert proof.complete is True
                assert values_equal(problem.expected_result, problem.metadata["target"])
            else:
                observed = LispEvaluator(stage=stage).evaluate_program(problem.prompt_expression)
                assert values_equal(observed, problem.expected_result)


@pytest.mark.parametrize(
    ("problem", "correct", "wrong", "eval_error", "garbage"),
    [
        (
            ProblemExample("(+ 1 2)", 3, DifficultyConfig(stage=1)),
            "3",
            "4",
            "(unknown)",
            "(+",
        ),
        (
            ProblemExample("(= 1 1)", parse_program(":true")[0], DifficultyConfig(stage=2)),
            ":true",
            "999",
            "(unknown)",
            "(+",
        ),
        (
            ProblemExample("(quote (+ 1 2))", parse_program("(+ 1 2)")[0], DifficultyConfig(stage=3)),
            "(+ 1 2)",
            "(* 9 9)",
            "x",
            "(+",
        ),
        (
            ProblemExample("(if (> 1 0) 2 3)", 2, DifficultyConfig(stage=4)),
            "2",
            "999",
            "(unknown)",
            "(+",
        ),
        (
            ProblemExample("(simplify (quote (+ x 0)))", parse_program("x")[0], DifficultyConfig(stage=5)),
            "x",
            "(- x 0)",
            "(unknown)",
            "(+",
        ),
        (
            ProblemExample("(count [1 2 3])", 3, DifficultyConfig(stage=7)),
            "3",
            "999",
            "(unknown)",
            "(+",
        ),
    ],
)
def test_phase_r0_reward_contract_for_evaluable_stages(
    problem: ProblemExample,
    correct: str,
    wrong: str,
    eval_error: str,
    garbage: str,
) -> None:
    reward = RewardFunction()

    assert reward.score_output(problem, correct).reward == 1.0
    assert reward.score_output(problem, wrong).reward == 0.5
    assert reward.score_output(problem, eval_error).reward == 0.2
    assert reward.score_output(problem, garbage).reward == 0.0


def test_phase_r0_reward_scores_stage_six_proofs() -> None:
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
    garbage = reward.score_output(problem, "(+")

    assert complete.reward == 1.0
    assert complete.error_type == "correct"
    assert invalid.reward < 1.0
    assert invalid.error_type == "invalid_proof_step"
    assert garbage.reward == 0.0
