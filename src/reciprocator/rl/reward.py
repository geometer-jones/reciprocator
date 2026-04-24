from __future__ import annotations

from dataclasses import dataclass

from .lisp_eval import (
    EvalError,
    Expression,
    InvalidProofStepError,
    ListExpr,
    LispEvaluator,
    ParseError,
    Symbol,
    WrongResultError,
    normalize_symbolic_expression,
    parse_program,
    render_value,
    same_shape,
    symbolic_equivalence,
    to_expression,
    validate_proof,
    values_equal,
)
from .problem_gen import ProblemExample


@dataclass(frozen=True)
class RewardResult:
    reward: float
    error_type: str
    parsed: bool
    evaluated: bool
    correct: bool
    actual_value: object | None
    expected_value: object

    @property
    def actual_value_text(self) -> str | None:
        return None if self.actual_value is None else render_value(self.actual_value)

    @property
    def expected_value_text(self) -> str:
        return render_value(self.expected_value)


class RewardFunction:
    def __init__(
        self,
        *,
        wrong_result_reward: float = 0.5,
        eval_error_reward: float = 0.2,
        stage_one_wrong_reward: float | None = None,
    ) -> None:
        self.wrong_result_reward = wrong_result_reward
        self.eval_error_reward = eval_error_reward
        self.stage_one_wrong_reward = stage_one_wrong_reward

    def score_output(self, problem: ProblemExample, output_text: str) -> RewardResult:
        from .lisp_eval import LispEvaluator

        candidate = output_text.strip()
        if not candidate:
            return RewardResult(0.0, "parse_error", parsed=False, evaluated=False, correct=False, actual_value=None, expected_value=problem.expected_result)

        stage = problem.difficulty.stage
        if stage == 3:
            return self._score_quote_stage(problem, candidate)
        if stage == 5:
            return self._score_symbolic_stage(problem, candidate)
        if stage == 6:
            return self._score_proof_stage(problem, candidate)

        evaluator = LispEvaluator(stage=stage)
        try:
            actual = evaluator.evaluate_program(candidate)
        except ParseError:
            return RewardResult(0.0, "parse_error", parsed=False, evaluated=False, correct=False, actual_value=None, expected_value=problem.expected_result)
        except EvalError:
            return RewardResult(self.eval_error_reward, "eval_error", parsed=True, evaluated=False, correct=False, actual_value=None, expected_value=problem.expected_result)

        if values_equal(actual, problem.expected_result):
            return RewardResult(1.0, "correct", parsed=True, evaluated=True, correct=True, actual_value=actual, expected_value=problem.expected_result)
        wrong_reward = self.stage_one_wrong_reward if stage == 1 and self.stage_one_wrong_reward is not None else self.wrong_result_reward
        return RewardResult(wrong_reward, "wrong_result", parsed=True, evaluated=True, correct=False, actual_value=actual, expected_value=problem.expected_result)

    def _score_quote_stage(self, problem: ProblemExample, candidate: str) -> RewardResult:
        try:
            forms = parse_program(candidate)
        except ParseError:
            return RewardResult(0.0, "parse_error", parsed=False, evaluated=False, correct=False, actual_value=None, expected_value=problem.expected_result)
        if not forms:
            return RewardResult(0.0, "parse_error", parsed=False, evaluated=False, correct=False, actual_value=None, expected_value=problem.expected_result)

        actual: object
        raw_form_is_data = (
            isinstance(problem.expected_result, ListExpr)
            and isinstance(forms[0], ListExpr)
            and not (
                forms[0].items
                and isinstance(forms[0].items[0], Symbol)
                and forms[0].items[0].name in {"quote", "list", "eval"}
            )
        )
        try:
            if raw_form_is_data:
                raise EvalError("Treating raw output list as data.")
            actual = LispEvaluator(stage=3).evaluate_program(candidate)
        except EvalError:
            if not raw_form_is_data:
                return RewardResult(
                    0.2,
                    "eval_error",
                    parsed=True,
                    evaluated=False,
                    correct=False,
                    actual_value=None,
                    expected_value=problem.expected_result,
                )
            actual = forms[0]

        if not isinstance(actual, ListExpr) and not isinstance(problem.expected_result, (int, float)):
            return RewardResult(0.0, "wrong_result", parsed=True, evaluated=True, correct=False, actual_value=actual, expected_value=problem.expected_result)
        if values_equal(actual, problem.expected_result):
            return RewardResult(1.0, "correct", parsed=True, evaluated=True, correct=True, actual_value=actual, expected_value=problem.expected_result)
        if isinstance(actual, (int, float)) and isinstance(problem.expected_result, (int, float)):
            return RewardResult(0.5, "wrong_result", parsed=True, evaluated=True, correct=False, actual_value=actual, expected_value=problem.expected_result)
        if isinstance(actual, ListExpr) and isinstance(problem.expected_result, ListExpr) and same_shape(actual, problem.expected_result):
            return RewardResult(0.5, "wrong_result", parsed=True, evaluated=True, correct=False, actual_value=actual, expected_value=problem.expected_result)
        return RewardResult(0.0, "wrong_result", parsed=True, evaluated=True, correct=False, actual_value=actual, expected_value=problem.expected_result)

    def _score_symbolic_stage(self, problem: ProblemExample, candidate: str) -> RewardResult:
        try:
            forms = parse_program(candidate)
        except ParseError:
            return RewardResult(0.0, "parse_error", parsed=False, evaluated=False, correct=False, actual_value=None, expected_value=problem.expected_result)

        try:
            actual = LispEvaluator(stage=5).evaluate_program(candidate)
            actual_expr = to_expression(actual)
        except EvalError:
            actual_expr = forms[0]

        expected_expr = to_expression(problem.expected_result)
        exact, numeric_score = symbolic_equivalence(actual_expr, expected_expr)
        if exact:
            return RewardResult(1.0, "correct", parsed=True, evaluated=True, correct=True, actual_value=actual_expr, expected_value=expected_expr)
        if numeric_score >= 0.75:
            return RewardResult(0.5, "wrong_result", parsed=True, evaluated=True, correct=False, actual_value=actual_expr, expected_value=expected_expr)
        return RewardResult(0.2, "parseable_expression", parsed=True, evaluated=True, correct=False, actual_value=actual_expr, expected_value=expected_expr)

    def _score_proof_stage(self, problem: ProblemExample, candidate: str) -> RewardResult:
        target = problem.metadata.get("target", problem.expected_result)
        premises = tuple(problem.metadata.get("premises", ()))
        try:
            proof = validate_proof(candidate, target=to_expression(target), premises=premises)
        except InvalidProofStepError:
            return RewardResult(0.0, "invalid_proof_step", parsed=True, evaluated=False, correct=False, actual_value=None, expected_value=problem.expected_result)
        reward = min(1.0, 0.8 * proof.valid_fraction + (0.2 if proof.complete else 0.0))
        return RewardResult(
            reward,
            "correct" if proof.complete else "invalid_proof_step",
            parsed=True,
            evaluated=True,
            correct=proof.complete,
            actual_value=proof,
            expected_value=problem.expected_result,
        )

    def score_outputs(self, problem: ProblemExample, outputs: list[str]) -> list[RewardResult]:
        return [self.score_output(problem, output) for output in outputs]


__all__ = [
    "RewardFunction",
    "RewardResult",
    "WrongResultError",
]
