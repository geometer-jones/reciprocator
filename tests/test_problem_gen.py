from reciprocator.rl.lisp_eval import LispEvaluator, values_equal
from reciprocator.rl.problem_gen import DifficultyConfig, ProblemGenerator, default_difficulty_for_stage


def test_problem_generator_stage_one_produces_evaluable_problem() -> None:
    generator = ProblemGenerator()
    difficulty = DifficultyConfig(stage=1, depth=2, operator_set=("+",), value_range=(0, 4))

    problem = generator.generate_problem(difficulty)
    observed = LispEvaluator(stage=1).evaluate_program(problem.prompt_expression)

    assert values_equal(observed, problem.expected_result)


def test_problem_generator_stage_six_returns_proof_target_and_premise_metadata() -> None:
    problem = ProblemGenerator().generate_problem(default_difficulty_for_stage(6))

    assert problem.prompt_expression.startswith("(prove ")
    assert "target" in problem.metadata
    assert "premises" in problem.metadata
    assert "example_solution" in problem.metadata
