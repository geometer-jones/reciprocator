import pytest

from reciprocator.rl.lisp_eval import (
    EvalError,
    LispEvaluator,
    Symbol,
    evaluate_program,
    parse_program,
    render_value,
    validate_proof,
)


def test_stage_one_evaluates_arithmetic_with_extra_ops() -> None:
    assert evaluate_program("(+ 1 2)", stage=1) == 3
    assert evaluate_program("(mod 8 3)", stage=1) == 2
    assert evaluate_program("(expt 2 3)", stage=1) == 8


def test_stage_two_supports_let_comparison_and_booleans() -> None:
    assert render_value(evaluate_program("(= 3 (+ 1 2))", stage=2)) == ":true"
    assert evaluate_program("(let [x (+ 3 4)] (* x 2))", stage=2) == 14
    assert render_value(evaluate_program("(and (> 3 2) (not (= 1 2)))", stage=2)) == ":true"


def test_stage_three_quote_list_and_eval_are_code_as_data() -> None:
    quoted = evaluate_program("(quote (+ 1 2))", stage=3)
    listed = evaluate_program("(list :eval (eval (quote (+ 1 2))))", stage=3)

    assert render_value(quoted) == "(+ 1 2)"
    assert render_value(listed) == "(:eval 3)"


def test_later_forms_are_stage_gated() -> None:
    with pytest.raises(EvalError):
        evaluate_program("(if :true 1 0)", stage=3)


def test_stage_four_supports_conditionals_and_functions() -> None:
    assert evaluate_program("(if (> 3 0) 4 0)", stage=4) == 4
    assert evaluate_program("(define double (lambda [x] (* x 2))) (double 7)", stage=4) == 14


def test_stage_five_symbolic_algebra_transforms_expressions() -> None:
    assert render_value(evaluate_program("(simplify (quote (+ x 0)))", stage=5)) == "x"
    assert render_value(evaluate_program("(differentiate (quote (* x x)) (quote x))", stage=5)) == "(* 2 x)"


def test_stage_six_validates_proof_steps() -> None:
    target = parse_program("x")[0]
    premise = parse_program("(+ x 0)")[0]

    proof = validate_proof(
        "(step simplify-add-zero (+ x 0) x)",
        target=target,
        premises=(premise,),
    )

    assert proof.complete is True
    assert proof.valid_fraction == 1.0


def test_stage_seven_supports_collections_and_threading() -> None:
    assert evaluate_program("(get [1 2 3] 1)", stage=7) == 2
    assert evaluate_program("(-> 3 (+ 4) (* 2))", stage=7) == 14
