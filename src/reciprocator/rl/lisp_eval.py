from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Callable, Optional, Union

Number = Union[int, float]
Expression = Union["Symbol", "Keyword", "ListExpr", "VectorExpr", "MapExpr", Number]


class LispError(Exception):
    """Base class for Lisp dialect failures."""


class ParseError(LispError):
    """Raised when text cannot be parsed as Lisp forms."""


class EvalError(LispError):
    """Raised when a parsed form cannot be evaluated."""


class WrongResultError(LispError):
    """Raised by callers that need a typed wrong-result failure."""


class InvalidProofStepError(LispError):
    """Raised when a proof step does not apply its claimed rewrite rule."""


@dataclass(frozen=True)
class Symbol:
    name: str


@dataclass(frozen=True)
class Keyword:
    name: str


@dataclass(frozen=True)
class ListExpr:
    items: tuple[Expression, ...]


@dataclass(frozen=True)
class VectorExpr:
    items: tuple[Expression, ...]


@dataclass(frozen=True)
class MapExpr:
    entries: tuple[tuple[Expression, Expression], ...]


@dataclass
class Environment:
    bindings: dict[str, object] = field(default_factory=dict)
    parent: Optional["Environment"] = None

    def lookup(self, symbol: Symbol | str) -> object:
        name = symbol if isinstance(symbol, str) else symbol.name
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise EvalError(f"Unknown symbol: {name}")

    def bind(self, name: str, value: object) -> None:
        self.bindings[name] = value


@dataclass
class Closure:
    params: tuple[str, ...]
    body: tuple[Expression, ...]
    env: Environment
    name: Optional[str] = None


@dataclass(frozen=True)
class BuiltinFunction:
    name: str
    fn: Callable[[list[object], tuple[str, ...]], object]


@dataclass(frozen=True)
class ProofStepResult:
    index: int
    rule_name: str
    valid: bool
    error: Optional[str]


@dataclass(frozen=True)
class ProofValidation:
    steps: tuple[ProofStepResult, ...]
    valid_fraction: float
    complete: bool


TRUE = Keyword("true")
FALSE = Keyword("false")
ELSE = Keyword("else")


def tokenize(source: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in source:
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current.clear()
            continue
        if char in "()[]{}":
            if current:
                tokens.append("".join(current))
                current.clear()
            tokens.append(char)
            continue
        current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _parse_atom(token: str) -> Expression:
    if token.startswith(":") and len(token) > 1:
        return Keyword(token[1:])
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)


def _parse(tokens: list[str], index: int = 0) -> tuple[Expression, int]:
    if index >= len(tokens):
        raise ParseError("Unexpected end of input.")
    token = tokens[index]
    if token == "(":
        items: list[Expression] = []
        cursor = index + 1
        while True:
            if cursor >= len(tokens):
                raise ParseError("Missing closing ')'.")
            if tokens[cursor] == ")":
                return ListExpr(tuple(items)), cursor + 1
            item, cursor = _parse(tokens, cursor)
            items.append(item)
    if token == "[":
        items: list[Expression] = []
        cursor = index + 1
        while True:
            if cursor >= len(tokens):
                raise ParseError("Missing closing ']'.")
            if tokens[cursor] == "]":
                return VectorExpr(tuple(items)), cursor + 1
            item, cursor = _parse(tokens, cursor)
            items.append(item)
    if token == "{":
        entries: list[tuple[Expression, Expression]] = []
        cursor = index + 1
        while True:
            if cursor >= len(tokens):
                raise ParseError("Missing closing '}'.")
            if tokens[cursor] == "}":
                return MapExpr(tuple(entries)), cursor + 1
            key, cursor = _parse(tokens, cursor)
            if cursor >= len(tokens) or tokens[cursor] == "}":
                raise ParseError("Map literals require key/value pairs.")
            value, cursor = _parse(tokens, cursor)
            entries.append((key, value))
    if token in ")]}":
        raise ParseError(f"Unexpected closing bracket: {token}")
    return _parse_atom(token), index + 1


def parse_program(source: str) -> list[Expression]:
    tokens = tokenize(source)
    if not tokens:
        raise ParseError("Expected at least one expression.")
    program: list[Expression] = []
    cursor = 0
    while cursor < len(tokens):
        form, cursor = _parse(tokens, cursor)
        program.append(form)
    return program


def _normalize_number(value: Number) -> Number:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def render_value(value: object) -> str:
    if isinstance(value, Symbol):
        return value.name
    if isinstance(value, Keyword):
        return f":{value.name}"
    if isinstance(value, ListExpr):
        return "(" + " ".join(render_value(item) for item in value.items) + ")"
    if isinstance(value, VectorExpr):
        return "[" + " ".join(render_value(item) for item in value.items) + "]"
    if isinstance(value, MapExpr):
        parts: list[str] = []
        for key, inner_value in value.entries:
            parts.append(render_value(key))
            parts.append(render_value(inner_value))
        return "{" + " ".join(parts) + "}"
    if isinstance(value, list):
        return "[" + " ".join(render_value(item) for item in value) + "]"
    if isinstance(value, dict):
        parts: list[str] = []
        for key, inner_value in value.items():
            parts.append(render_value(key))
            parts.append(render_value(inner_value))
        return "{" + " ".join(parts) + "}"
    if isinstance(value, Closure):
        return f"<fn:{value.name or 'lambda'}>"
    if isinstance(value, BuiltinFunction):
        return f"<builtin:{value.name}>"
    if isinstance(value, float):
        return str(_normalize_number(value))
    return str(value)


def values_equal(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-6
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(values_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (ListExpr, VectorExpr, MapExpr, Symbol, Keyword)) or isinstance(
        right, (ListExpr, VectorExpr, MapExpr, Symbol, Keyword)
    ):
        return render_value(left) == render_value(right)
    return left == right


def same_shape(left: Expression, right: Expression) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, (Symbol, Keyword)):
        return True
    if isinstance(left, ListExpr):
        return len(left.items) == len(right.items) and all(same_shape(a, b) for a, b in zip(left.items, right.items))
    if isinstance(left, VectorExpr):
        return len(left.items) == len(right.items) and all(same_shape(a, b) for a, b in zip(left.items, right.items))
    if isinstance(left, MapExpr):
        return len(left.entries) == len(right.entries)
    return False


def to_expression(value: object) -> Expression:
    if isinstance(value, (int, float, Symbol, Keyword, ListExpr, VectorExpr, MapExpr)):
        return value
    if isinstance(value, list):
        return VectorExpr(tuple(to_expression(item) for item in value))
    if isinstance(value, dict):
        return MapExpr(tuple((to_expression(key), to_expression(inner_value)) for key, inner_value in value.items()))
    raise EvalError(f"Cannot treat runtime value as code: {render_value(value)}")


def _is_symbol(expr: Expression, name: str) -> bool:
    return isinstance(expr, Symbol) and expr.name == name


def _sorted_exprs(items: list[Expression]) -> list[Expression]:
    return sorted(items, key=render_value)


def simplify_expression(expr: Expression) -> Expression:
    if isinstance(expr, (int, float, Symbol, Keyword)):
        return expr
    if isinstance(expr, VectorExpr):
        return VectorExpr(tuple(simplify_expression(item) for item in expr.items))
    if isinstance(expr, MapExpr):
        return MapExpr(tuple((simplify_expression(key), simplify_expression(value)) for key, value in expr.entries))
    if not expr.items:
        return expr
    head = expr.items[0]
    args = [simplify_expression(item) for item in expr.items[1:]]
    if _is_symbol(head, "+"):
        flattened: list[Expression] = []
        numeric_total = 0.0
        for argument in args:
            if isinstance(argument, (int, float)):
                numeric_total += float(argument)
            elif isinstance(argument, ListExpr) and argument.items and _is_symbol(argument.items[0], "+"):
                flattened.extend(argument.items[1:])
            else:
                flattened.append(argument)
        if abs(numeric_total) > 1e-6:
            flattened.append(_normalize_number(numeric_total))
        flattened = [item for item in flattened if not (isinstance(item, (int, float)) and abs(float(item)) <= 1e-6)]
        flattened = _sorted_exprs(flattened)
        if not flattened:
            return 0
        if len(flattened) == 1:
            return flattened[0]
        return ListExpr((Symbol("+"), *flattened))
    if _is_symbol(head, "*"):
        flattened = []
        numeric_total = 1.0
        for argument in args:
            if isinstance(argument, (int, float)):
                numeric_total *= float(argument)
            elif isinstance(argument, ListExpr) and argument.items and _is_symbol(argument.items[0], "*"):
                flattened.extend(argument.items[1:])
            else:
                flattened.append(argument)
        if abs(numeric_total) <= 1e-6:
            return 0
        if abs(numeric_total - 1.0) > 1e-6 or not flattened:
            flattened.append(_normalize_number(numeric_total))
        flattened = [item for item in flattened if not (isinstance(item, (int, float)) and abs(float(item) - 1.0) <= 1e-6)]
        flattened = _sorted_exprs(flattened)
        if not flattened:
            return 1
        if len(flattened) == 1:
            return flattened[0]
        return ListExpr((Symbol("*"), *flattened))
    if _is_symbol(head, "-") and len(args) == 2 and all(isinstance(arg, (int, float)) for arg in args):
        return _normalize_number(float(args[0]) - float(args[1]))
    if _is_symbol(head, "mod") and len(args) == 2 and all(isinstance(arg, (int, float)) for arg in args):
        return int(args[0]) % int(args[1])
    if _is_symbol(head, "expt") and len(args) == 2:
        base, power = args
        if isinstance(power, (int, float)):
            if abs(float(power)) <= 1e-6:
                return 1
            if abs(float(power) - 1.0) <= 1e-6:
                return base
            if isinstance(base, (int, float)):
                return _normalize_number(float(base) ** float(power))
        return ListExpr((Symbol("expt"), base, power))
    return ListExpr((head, *args))


def expand_expression(expr: Expression) -> Expression:
    expr = simplify_expression(expr)
    if not isinstance(expr, ListExpr) or not expr.items:
        return expr
    head = expr.items[0]
    args = [expand_expression(item) for item in expr.items[1:]]
    if _is_symbol(head, "*") and len(args) == 2:
        left, right = args
        if isinstance(left, ListExpr) and left.items and _is_symbol(left.items[0], "+"):
            terms = [expand_expression(ListExpr((Symbol("*"), term, right))) for term in left.items[1:]]
            return simplify_expression(ListExpr((Symbol("+"), *terms)))
        if isinstance(right, ListExpr) and right.items and _is_symbol(right.items[0], "+"):
            terms = [expand_expression(ListExpr((Symbol("*"), left, term))) for term in right.items[1:]]
            return simplify_expression(ListExpr((Symbol("+"), *terms)))
    return simplify_expression(ListExpr((head, *args)))


def substitute_expression(expr: Expression, target: Expression, replacement: Expression) -> Expression:
    if values_equal(expr, target):
        return replacement
    if isinstance(expr, ListExpr):
        return ListExpr(tuple(substitute_expression(item, target, replacement) for item in expr.items))
    if isinstance(expr, VectorExpr):
        return VectorExpr(tuple(substitute_expression(item, target, replacement) for item in expr.items))
    if isinstance(expr, MapExpr):
        return MapExpr(
            tuple(
                (substitute_expression(key, target, replacement), substitute_expression(value, target, replacement))
                for key, value in expr.entries
            )
        )
    return expr


def differentiate_expression(expr: Expression, variable: Symbol) -> Expression:
    if isinstance(expr, (int, float, Keyword)):
        return 0
    if isinstance(expr, Symbol):
        return 1 if expr.name == variable.name else 0
    if isinstance(expr, (VectorExpr, MapExpr)):
        raise EvalError("differentiate expects an arithmetic expression.")
    if not expr.items:
        raise EvalError("Cannot differentiate an empty list.")
    head = expr.items[0]
    args = expr.items[1:]
    if _is_symbol(head, "+"):
        return simplify_expression(ListExpr((Symbol("+"), *(differentiate_expression(arg, variable) for arg in args))))
    if _is_symbol(head, "-"):
        if len(args) == 2:
            return simplify_expression(
                ListExpr((Symbol("-"), differentiate_expression(args[0], variable), differentiate_expression(args[1], variable)))
            )
    if _is_symbol(head, "*") and len(args) == 2:
        left, right = args
        if values_equal(left, right):
            return simplify_expression(ListExpr((Symbol("*"), 2, left)))
        return simplify_expression(
            ListExpr(
                (
                    Symbol("+"),
                    ListExpr((Symbol("*"), differentiate_expression(left, variable), right)),
                    ListExpr((Symbol("*"), left, differentiate_expression(right, variable))),
                )
            )
        )
    if _is_symbol(head, "expt") and len(args) == 2 and isinstance(args[1], (int, float)):
        base, power = args
        return simplify_expression(
            ListExpr(
                (
                    Symbol("*"),
                    power,
                    ListExpr((Symbol("*"), ListExpr((Symbol("expt"), base, power - 1)), differentiate_expression(base, variable))),
                )
            )
        )
    raise EvalError("Unsupported differentiation form.")


def collect_free_symbols(expr: Expression) -> set[str]:
    if isinstance(expr, Symbol):
        if expr.name in {
            "+",
            "-",
            "*",
            "/",
            "mod",
            "expt",
            "=",
            "<",
            ">",
            "<=",
            ">=",
            "and",
            "or",
            "not",
            "quote",
            "list",
            "eval",
            "let",
            "if",
            "cond",
            "define",
            "lambda",
            "simplify",
            "expand",
            "substitute",
            "differentiate",
        }:
            return set()
        return {expr.name}
    if isinstance(expr, ListExpr):
        found: set[str] = set()
        for item in expr.items:
            found.update(collect_free_symbols(item))
        return found
    if isinstance(expr, VectorExpr):
        found = set()
        for item in expr.items:
            found.update(collect_free_symbols(item))
        return found
    if isinstance(expr, MapExpr):
        found = set()
        for key, value in expr.entries:
            found.update(collect_free_symbols(key))
            found.update(collect_free_symbols(value))
        return found
    return set()


def normalize_symbolic_expression(expr: Expression) -> Expression:
    return simplify_expression(expand_expression(expr))


def symbolic_equivalence(left: Expression, right: Expression, *, samples: int = 8, seed: int = 0) -> tuple[bool, float]:
    normalized_left = normalize_symbolic_expression(left)
    normalized_right = normalize_symbolic_expression(right)
    if render_value(normalized_left) == render_value(normalized_right):
        return True, 1.0

    variables = sorted(collect_free_symbols(normalized_left) | collect_free_symbols(normalized_right))
    if not variables:
        return False, 0.0
    rng = random.Random(seed)
    matches = 0
    for _ in range(samples):
        assignments = {name: rng.randint(-3, 3) or 1 for name in variables}
        if _evaluate_symbolic_numeric(normalized_left, assignments) == _evaluate_symbolic_numeric(normalized_right, assignments):
            matches += 1
    return False, matches / samples


def _evaluate_symbolic_numeric(expr: Expression, assignments: dict[str, Number]) -> object:
    evaluator = LispEvaluator(stage=2)
    env = evaluator._base_env()
    for name, value in assignments.items():
        env.bind(name, value)
    return evaluator._eval(expr, env, (), in_quote=False)


class LispEvaluator:
    def __init__(self, stage: int = 1, *, max_call_depth: int = 64) -> None:
        if not 1 <= stage <= 7:
            raise ValueError("stage must be in the range [1, 7].")
        self.stage = stage
        self.max_call_depth = max_call_depth

    def evaluate_program(self, source: str, env: Optional[Environment] = None) -> object:
        scope = env or self._base_env()
        result: object = None
        for form in parse_program(source):
            result = self._eval(form, scope, (), in_quote=False)
        return result

    def _base_env(self) -> Environment:
        env = Environment()

        def register(name: str, fn: Callable[[list[object], tuple[str, ...]], object], *, min_stage: int = 1) -> None:
            if self.stage >= min_stage:
                env.bind(name, BuiltinFunction(name=name, fn=fn))

        register("+", lambda args, _: _normalize_number(sum(self._num(arg) for arg in args)), min_stage=1)
        register("-", self._subtract, min_stage=1)
        register("*", self._multiply, min_stage=1)
        register("/", self._divide, min_stage=1)
        register("mod", self._mod, min_stage=1)
        register("expt", self._expt, min_stage=1)
        register("=", lambda args, _: TRUE if self._all_equal(args) else FALSE, min_stage=2)
        register("<", lambda args, _: TRUE if self._ordered(args, lambda a, b: a < b) else FALSE, min_stage=2)
        register(">", lambda args, _: TRUE if self._ordered(args, lambda a, b: a > b) else FALSE, min_stage=2)
        register("<=", lambda args, _: TRUE if self._ordered(args, lambda a, b: a <= b) else FALSE, min_stage=2)
        register(">=", lambda args, _: TRUE if self._ordered(args, lambda a, b: a >= b) else FALSE, min_stage=2)
        register("and", lambda args, _: TRUE if all(self._truthy(arg) for arg in args) else FALSE, min_stage=2)
        register("or", lambda args, _: TRUE if any(self._truthy(arg) for arg in args) else FALSE, min_stage=2)
        register("not", lambda args, _: TRUE if len(args) == 1 and not self._truthy(args[0]) else FALSE, min_stage=2)
        register("list", self._list_builtin, min_stage=3)
        register("eval", self._eval_builtin, min_stage=3)
        register("simplify", self._simplify_builtin, min_stage=5)
        register("expand", self._expand_builtin, min_stage=5)
        register("substitute", self._substitute_builtin, min_stage=5)
        register("differentiate", self._differentiate_builtin, min_stage=5)
        register("get", self._get, min_stage=7)
        register("count", self._count, min_stage=7)
        register("conj", self._conj, min_stage=7)
        register("assoc", self._assoc, min_stage=7)
        register("mapv", self._mapv, min_stage=7)
        register("filterv", self._filterv, min_stage=7)
        register("reduce", self._reduce, min_stage=7)
        register("comp", self._comp, min_stage=7)
        register("partial", self._partial, min_stage=7)
        return env

    def _eval(self, expr: Expression, env: Environment, call_stack: tuple[str, ...], *, in_quote: bool) -> object:
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, Keyword):
            return expr
        if isinstance(expr, Symbol):
            return expr if in_quote else env.lookup(expr)
        if isinstance(expr, VectorExpr):
            if in_quote:
                return expr
            if self.stage < 7:
                raise EvalError("Vector values unlock at stage 7.")
            return [self._eval(item, env, call_stack, in_quote=False) for item in expr.items]
        if isinstance(expr, MapExpr):
            if in_quote:
                return expr
            if self.stage < 7:
                raise EvalError("Map values unlock at stage 7.")
            return {
                self._eval(key, env, call_stack, in_quote=False): self._eval(value, env, call_stack, in_quote=False)
                for key, value in expr.entries
            }

        if not expr.items:
            raise EvalError("Cannot evaluate an empty list.")
        head = expr.items[0]
        if isinstance(head, Symbol):
            if head.name == "quote":
                self._require_stage(3, "quote")
                if len(expr.items) != 2:
                    raise EvalError("quote expects exactly one argument.")
                return expr.items[1]
            if head.name == "let":
                self._require_stage(2, "let")
                return self._eval_let(expr, env, call_stack)
            if head.name == "if":
                self._require_stage(4, "if")
                return self._eval_if(expr, env, call_stack)
            if head.name == "cond":
                self._require_stage(4, "cond")
                return self._eval_cond(expr, env, call_stack)
            if head.name == "define":
                self._require_stage(4, "define")
                return self._eval_define(expr, env, call_stack)
            if head.name == "lambda":
                self._require_stage(4, "lambda")
                return self._eval_lambda(expr, env)
            if head.name in {"->", "->>"}:
                self._require_stage(7, head.name)
                return self._eval_threading(expr, env, call_stack, insert_last=(head.name == "->>"))

        procedure = self._eval(head, env, call_stack, in_quote=False)
        arguments = [self._eval(item, env, call_stack, in_quote=False) for item in expr.items[1:]]
        return self._invoke(procedure, arguments, call_stack)

    def _eval_sequence(self, forms: tuple[Expression, ...], env: Environment, call_stack: tuple[str, ...]) -> object:
        if not forms:
            raise EvalError("Expected at least one form.")
        result: object = None
        for form in forms:
            result = self._eval(form, env, call_stack, in_quote=False)
        return result

    def _eval_let(self, expr: ListExpr, env: Environment, call_stack: tuple[str, ...]) -> object:
        if len(expr.items) < 3 or not isinstance(expr.items[1], VectorExpr):
            raise EvalError("let expects (let [name value ...] body...).")
        bindings = expr.items[1].items
        if len(bindings) % 2 != 0:
            raise EvalError("let bindings must contain name/value pairs.")
        local = Environment(parent=env)
        for index in range(0, len(bindings), 2):
            name = bindings[index]
            if not isinstance(name, Symbol):
                raise EvalError("let binding names must be symbols.")
            local.bind(name.name, self._eval(bindings[index + 1], local, call_stack, in_quote=False))
        return self._eval_sequence(expr.items[2:], local, call_stack)

    def _eval_if(self, expr: ListExpr, env: Environment, call_stack: tuple[str, ...]) -> object:
        if len(expr.items) != 4:
            raise EvalError("if expects (if test then else).")
        condition = self._eval(expr.items[1], env, call_stack, in_quote=False)
        branch = expr.items[2] if self._truthy(condition) else expr.items[3]
        return self._eval(branch, env, call_stack, in_quote=False)

    def _eval_cond(self, expr: ListExpr, env: Environment, call_stack: tuple[str, ...]) -> object:
        clauses = expr.items[1:]
        if len(clauses) % 2 != 0:
            raise EvalError("cond expects test/result pairs.")
        for index in range(0, len(clauses), 2):
            test = clauses[index]
            if isinstance(test, Keyword) and test == ELSE:
                return self._eval(clauses[index + 1], env, call_stack, in_quote=False)
            if self._truthy(self._eval(test, env, call_stack, in_quote=False)):
                return self._eval(clauses[index + 1], env, call_stack, in_quote=False)
        return None

    def _eval_define(self, expr: ListExpr, env: Environment, call_stack: tuple[str, ...]) -> object:
        if len(expr.items) != 3 or not isinstance(expr.items[1], Symbol):
            raise EvalError("define expects (define name expr).")
        value = self._eval(expr.items[2], env, call_stack, in_quote=False)
        env.bind(expr.items[1].name, value)
        if isinstance(value, Closure):
            value.name = expr.items[1].name
        return value

    def _eval_lambda(self, expr: ListExpr, env: Environment) -> Closure:
        if len(expr.items) < 3 or not isinstance(expr.items[1], VectorExpr):
            raise EvalError("lambda expects (lambda [params] body...).")
        params = []
        for item in expr.items[1].items:
            if not isinstance(item, Symbol):
                raise EvalError("lambda parameters must be symbols.")
            params.append(item.name)
        return Closure(params=tuple(params), body=expr.items[2:], env=env)

    def _eval_threading(self, expr: ListExpr, env: Environment, call_stack: tuple[str, ...], *, insert_last: bool) -> object:
        value = self._eval(expr.items[1], env, call_stack, in_quote=False)
        for step in expr.items[2:]:
            if isinstance(step, ListExpr):
                procedure = self._eval(step.items[0], env, call_stack, in_quote=False)
                arguments = [self._eval(item, env, call_stack, in_quote=False) for item in step.items[1:]]
                arguments = arguments + [value] if insert_last else [value] + arguments
                value = self._invoke(procedure, arguments, call_stack)
            else:
                procedure = self._eval(step, env, call_stack, in_quote=False)
                value = self._invoke(procedure, [value], call_stack)
        return value

    def _invoke(self, procedure: object, args: list[object], call_stack: tuple[str, ...]) -> object:
        if isinstance(procedure, BuiltinFunction):
            return procedure.fn(args, call_stack)
        if isinstance(procedure, Closure):
            if len(args) != len(procedure.params):
                raise EvalError(f"{procedure.name or 'lambda'} expected {len(procedure.params)} args, received {len(args)}.")
            if len(call_stack) >= self.max_call_depth:
                raise EvalError("Maximum call depth exceeded.")
            local = Environment(parent=procedure.env)
            for name, value in zip(procedure.params, args):
                local.bind(name, value)
            return self._eval_sequence(procedure.body, local, call_stack + (procedure.name or "<lambda>",))
        raise EvalError(f"Attempted to call non-function value: {render_value(procedure)}")

    def _require_stage(self, minimum_stage: int, feature: str) -> None:
        if self.stage < minimum_stage:
            raise EvalError(f"{feature} unlocks at stage {minimum_stage}.")

    @staticmethod
    def _num(value: object) -> Number:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvalError(f"Expected a number, received {render_value(value)}.")
        return value

    @staticmethod
    def _truthy(value: object) -> bool:
        if value is None:
            return False
        if value == FALSE:
            return False
        if isinstance(value, (int, float)) and value == 0:
            return False
        if value == [] or value == {}:
            return False
        return True

    def _subtract(self, args: list[object], _: tuple[str, ...]) -> Number:
        if not args:
            raise EvalError("- expects at least one argument.")
        result = self._num(args[0])
        if len(args) == 1:
            return _normalize_number(-result)
        for value in args[1:]:
            result -= self._num(value)
        return _normalize_number(result)

    def _multiply(self, args: list[object], _: tuple[str, ...]) -> Number:
        result: Number = 1
        for value in args:
            result *= self._num(value)
        return _normalize_number(result)

    def _divide(self, args: list[object], _: tuple[str, ...]) -> Number:
        if not args:
            raise EvalError("/ expects at least one argument.")
        result = float(self._num(args[0]))
        for value in args[1:]:
            divisor = self._num(value)
            if divisor == 0:
                raise EvalError("Division by zero.")
            result /= divisor
        return _normalize_number(result)

    def _mod(self, args: list[object], _: tuple[str, ...]) -> int:
        if len(args) != 2:
            raise EvalError("mod expects exactly two arguments.")
        return int(self._num(args[0])) % int(self._num(args[1]))

    def _expt(self, args: list[object], _: tuple[str, ...]) -> Number:
        if len(args) != 2:
            raise EvalError("expt expects exactly two arguments.")
        return _normalize_number(self._num(args[0]) ** self._num(args[1]))

    @staticmethod
    def _all_equal(values: list[object]) -> bool:
        return all(values_equal(values[0], value) for value in values[1:]) if values else True

    def _ordered(self, values: list[object], predicate: Callable[[Number, Number], bool]) -> bool:
        if len(values) < 2:
            raise EvalError("Comparison operators expect at least two arguments.")
        numbers = [self._num(value) for value in values]
        return all(predicate(numbers[index], numbers[index + 1]) for index in range(len(numbers) - 1))

    def _list_builtin(self, args: list[object], _: tuple[str, ...]) -> ListExpr:
        return ListExpr(tuple(to_expression(argument) for argument in args))

    def _eval_builtin(self, args: list[object], call_stack: tuple[str, ...]) -> object:
        if len(args) != 1:
            raise EvalError("eval expects exactly one argument.")
        return self._eval(to_expression(args[0]), self._base_env(), call_stack, in_quote=False)

    def _simplify_builtin(self, args: list[object], _: tuple[str, ...]) -> Expression:
        if len(args) != 1:
            raise EvalError("simplify expects exactly one argument.")
        return normalize_symbolic_expression(to_expression(args[0]))

    def _expand_builtin(self, args: list[object], _: tuple[str, ...]) -> Expression:
        if len(args) != 1:
            raise EvalError("expand expects exactly one argument.")
        return expand_expression(to_expression(args[0]))

    def _substitute_builtin(self, args: list[object], _: tuple[str, ...]) -> Expression:
        if len(args) != 3:
            raise EvalError("substitute expects (substitute expr target replacement).")
        return normalize_symbolic_expression(
            substitute_expression(
                to_expression(args[0]),
                to_expression(args[1]),
                to_expression(args[2]),
            )
        )

    def _differentiate_builtin(self, args: list[object], _: tuple[str, ...]) -> Expression:
        if len(args) != 2 or not isinstance(to_expression(args[1]), Symbol):
            raise EvalError("differentiate expects (differentiate expr symbol).")
        return normalize_symbolic_expression(differentiate_expression(to_expression(args[0]), to_expression(args[1])))

    def _get(self, args: list[object], _: tuple[str, ...]) -> object:
        if len(args) not in {2, 3}:
            raise EvalError("get expects (get coll key [default]).")
        coll, key = args[0], args[1]
        default = args[2] if len(args) == 3 else None
        if isinstance(coll, list):
            if not isinstance(key, int):
                raise EvalError("Vector indices must be integers.")
            return coll[key] if 0 <= key < len(coll) else default
        if isinstance(coll, dict):
            return coll.get(key, default)
        raise EvalError("get expects a vector or map.")

    def _count(self, args: list[object], _: tuple[str, ...]) -> int:
        if len(args) != 1 or not isinstance(args[0], (list, dict)):
            raise EvalError("count expects a single vector or map.")
        return len(args[0])

    def _conj(self, args: list[object], _: tuple[str, ...]) -> object:
        if len(args) < 2:
            raise EvalError("conj expects at least two arguments.")
        coll = args[0]
        if isinstance(coll, list):
            return list(coll) + list(args[1:])
        if isinstance(coll, dict):
            updated = dict(coll)
            for value in args[1:]:
                if isinstance(value, dict):
                    updated.update(value)
                elif isinstance(value, list) and len(value) == 2:
                    updated[value[0]] = value[1]
                else:
                    raise EvalError("Map conj expects maps or [key value] vectors.")
            return updated
        raise EvalError("conj expects a vector or map.")

    def _assoc(self, args: list[object], _: tuple[str, ...]) -> object:
        if len(args) < 3 or len(args[1:]) % 2 != 0:
            raise EvalError("assoc expects (assoc coll key value ...).")
        coll = args[0]
        if isinstance(coll, dict):
            updated = dict(coll)
            for index in range(1, len(args), 2):
                updated[args[index]] = args[index + 1]
            return updated
        if isinstance(coll, list):
            updated = list(coll)
            for index in range(1, len(args), 2):
                position = args[index]
                if not isinstance(position, int) or not 0 <= position < len(updated):
                    raise EvalError("Vector assoc indices must be in range.")
                updated[position] = args[index + 1]
            return updated
        raise EvalError("assoc expects a vector or map.")

    def _mapv(self, args: list[object], call_stack: tuple[str, ...]) -> list[object]:
        if len(args) != 2 or not isinstance(args[1], list):
            raise EvalError("mapv expects (mapv fn vector).")
        return [self._invoke(args[0], [item], call_stack) for item in args[1]]

    def _filterv(self, args: list[object], call_stack: tuple[str, ...]) -> list[object]:
        if len(args) != 2 or not isinstance(args[1], list):
            raise EvalError("filterv expects (filterv fn vector).")
        return [item for item in args[1] if self._truthy(self._invoke(args[0], [item], call_stack))]

    def _reduce(self, args: list[object], call_stack: tuple[str, ...]) -> object:
        if len(args) not in {2, 3}:
            raise EvalError("reduce expects (reduce fn vector) or (reduce fn init vector).")
        fn = args[0]
        if len(args) == 2:
            if not isinstance(args[1], list) or not args[1]:
                raise EvalError("reduce with two arguments expects a non-empty vector.")
            accumulator = args[1][0]
            items = args[1][1:]
        else:
            accumulator = args[1]
            if not isinstance(args[2], list):
                raise EvalError("reduce expects a vector in the final position.")
            items = args[2]
        for item in items:
            accumulator = self._invoke(fn, [accumulator, item], call_stack)
        return accumulator

    def _comp(self, args: list[object], _: tuple[str, ...]) -> BuiltinFunction:
        functions = list(args)

        def composed(inner_args: list[object], call_stack: tuple[str, ...]) -> object:
            value = self._invoke(functions[-1], inner_args, call_stack)
            for fn in reversed(functions[:-1]):
                value = self._invoke(fn, [value], call_stack)
            return value

        return BuiltinFunction(name="comp", fn=composed)

    def _partial(self, args: list[object], _: tuple[str, ...]) -> BuiltinFunction:
        if not args:
            raise EvalError("partial expects at least one argument.")
        fn = args[0]
        prefix = list(args[1:])

        def partially_applied(inner_args: list[object], call_stack: tuple[str, ...]) -> object:
            return self._invoke(fn, prefix + list(inner_args), call_stack)

        return BuiltinFunction(name="partial", fn=partially_applied)


def _proof_rule_output(rule_name: str, input_expr: Expression) -> Expression:
    normalized_input = normalize_symbolic_expression(input_expr)
    if rule_name == "simplify-add-zero":
        return simplify_expression(normalized_input)
    if rule_name == "simplify-mul-one":
        return simplify_expression(normalized_input)
    if rule_name == "simplify-mul-zero":
        return simplify_expression(normalized_input)
    if rule_name == "commute-add" and isinstance(normalized_input, ListExpr) and _is_symbol(normalized_input.items[0], "+") and len(normalized_input.items) == 3:
        return ListExpr((Symbol("+"), normalized_input.items[2], normalized_input.items[1]))
    if rule_name == "commute-mul" and isinstance(normalized_input, ListExpr) and _is_symbol(normalized_input.items[0], "*") and len(normalized_input.items) == 3:
        return ListExpr((Symbol("*"), normalized_input.items[2], normalized_input.items[1]))
    if rule_name == "distribute":
        return expand_expression(normalized_input)
    raise InvalidProofStepError(f"Unknown or inapplicable proof rule: {rule_name}")


def validate_proof(
    proof_source: str,
    *,
    target: Expression,
    premises: tuple[Expression, ...] = (),
) -> ProofValidation:
    try:
        forms = parse_program(proof_source)
    except ParseError as exc:
        raise InvalidProofStepError(str(exc)) from exc
    if not forms:
        return ProofValidation(steps=(), valid_fraction=0.0, complete=False)

    step_results: list[ProofStepResult] = []
    previous_output: Optional[Expression] = None
    valid_count = 0
    for index, form in enumerate(forms):
        if not isinstance(form, ListExpr) or len(form.items) != 4 or not _is_symbol(form.items[0], "step") or not isinstance(form.items[1], Symbol):
            step_results.append(ProofStepResult(index=index, rule_name="<invalid>", valid=False, error="Malformed step form."))
            continue
        rule_name = form.items[1].name
        input_expr = form.items[2]
        output_expr = form.items[3]
        try:
            if previous_output is None and premises and not any(values_equal(input_expr, premise) for premise in premises):
                raise InvalidProofStepError("First step input does not match any premise.")
            if previous_output is not None and not values_equal(input_expr, previous_output):
                raise InvalidProofStepError("Step input does not match previous step output.")
            expected_output = _proof_rule_output(rule_name, input_expr)
            if not values_equal(normalize_symbolic_expression(output_expr), normalize_symbolic_expression(expected_output)):
                raise InvalidProofStepError("Step output does not match the claimed rule.")
            valid_count += 1
            previous_output = output_expr
            step_results.append(ProofStepResult(index=index, rule_name=rule_name, valid=True, error=None))
        except InvalidProofStepError as exc:
            step_results.append(ProofStepResult(index=index, rule_name=rule_name, valid=False, error=str(exc)))
            previous_output = output_expr

    valid_fraction = valid_count / len(forms)
    complete = valid_count == len(forms) and previous_output is not None and symbolic_equivalence(previous_output, target)[0]
    return ProofValidation(steps=tuple(step_results), valid_fraction=valid_fraction, complete=complete)


def evaluate_program(source: str, *, stage: int = 1) -> object:
    return LispEvaluator(stage=stage).evaluate_program(source)


__all__ = [
    "BuiltinFunction",
    "Closure",
    "ELSE",
    "Environment",
    "EvalError",
    "Expression",
    "FALSE",
    "InvalidProofStepError",
    "Keyword",
    "ListExpr",
    "LispEvaluator",
    "LispError",
    "MapExpr",
    "ParseError",
    "ProofStepResult",
    "ProofValidation",
    "Symbol",
    "TRUE",
    "VectorExpr",
    "WrongResultError",
    "collect_free_symbols",
    "differentiate_expression",
    "evaluate_program",
    "expand_expression",
    "normalize_symbolic_expression",
    "parse_program",
    "render_value",
    "same_shape",
    "simplify_expression",
    "substitute_expression",
    "symbolic_equivalence",
    "tokenize",
    "to_expression",
    "validate_proof",
    "values_equal",
]
