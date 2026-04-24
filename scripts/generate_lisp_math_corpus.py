#!/usr/bin/env python3
"""Generate a deterministic Lisp-math pretraining corpus.

Each example is formatted as:

    <prompt expression>
    <expected answer or proof>

The prompt and answer are both parseable by the local Lisp parser. For stages
1-5 and 7, answers are computed by the evaluator. For stage 6, proof answers
are checked by the proof validator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reciprocator.rl.lisp_eval import (  # noqa: E402
    LispEvaluator,
    parse_program,
    render_value,
    validate_proof,
)


CORPUS_NAME = "lisp_math"
COMBINED_FILENAME = "lisp_math_combined.txt"
DEFAULT_TARGET_CHARS = 10_000_000
DEFAULT_SEED = 20260423
STAGE_WEIGHTS = (
    (1, 24),
    (2, 18),
    (3, 12),
    (4, 14),
    (5, 16),
    (6, 8),
    (7, 8),
)


@dataclass(frozen=True)
class CorpusExample:
    stage: int
    kind: str
    prompt: str
    answer: str

    @property
    def block(self) -> str:
        return f"{self.prompt}\n{self.answer}\n\n"


class LispMathExampleGenerator:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def generate(self) -> CorpusExample:
        stages, weights = zip(*STAGE_WEIGHTS)
        stage = self.rng.choices(stages, weights=weights, k=1)[0]
        generators: dict[int, Callable[[], CorpusExample]] = {
            1: self.stage_one,
            2: self.stage_two,
            3: self.stage_three,
            4: self.stage_four,
            5: self.stage_five,
            6: self.stage_six,
            7: self.stage_seven,
        }
        return generators[stage]()

    def literal(self, low: int = 0, high: int = 9, *, non_zero: bool = False) -> int:
        value = self.rng.randint(low, high)
        if non_zero and value == 0:
            candidates = [candidate for candidate in range(low, high + 1) if candidate != 0]
            value = self.rng.choice(candidates) if candidates else 1
        return value

    def arithmetic(self, depth: int, *, low: int = -9, high: int = 9) -> str:
        if depth <= 1:
            return str(self.literal(low, high))

        operator = self.rng.choice(("+", "-", "*", "/", "mod", "expt"))
        if operator == "/":
            divisor = self.literal(1, 9, non_zero=True)
            quotient = self.literal(low, high)
            return f"(/ {quotient * divisor} {divisor})"
        if operator == "mod":
            divisor = self.literal(1, 9, non_zero=True)
            remainder = self.rng.randrange(divisor)
            multiplier = self.literal(0, 6)
            return f"(mod {multiplier * divisor + remainder} {divisor})"
        if operator == "expt":
            base = self.literal(0, 6)
            power = self.literal(0, 3)
            return f"(expt {base} {power})"

        arity = 2 if operator in {"-", "*"} else self.rng.randint(2, 4)
        args = [self.arithmetic(depth - 1, low=low, high=high) for _ in range(arity)]
        return f"({operator} {' '.join(args)})"

    def evaluated_example(self, stage: int, kind: str, prompt: str) -> CorpusExample:
        expected = LispEvaluator(stage=stage).evaluate_program(prompt)
        answer = render_value(expected)
        parse_program(prompt)
        parse_program(answer)
        return CorpusExample(stage=stage, kind=kind, prompt=prompt, answer=answer)

    def stage_one(self) -> CorpusExample:
        depth = self.rng.choices((1, 2, 3, 4), weights=(2, 5, 5, 2), k=1)[0]
        return self.evaluated_example(1, "arithmetic", self.arithmetic(depth))

    def stage_two(self) -> CorpusExample:
        choice = self.rng.choice(("let-one", "let-two", "compare", "boolean"))
        if choice == "let-one":
            x_expr = self.arithmetic(self.rng.randint(1, 3), low=0, high=9)
            body = self.rng.choice((
                f"(* x {self.literal(1, 5)})",
                f"(+ x {self.arithmetic(2, low=0, high=5)})",
                f"(- x {self.literal(0, 5)})",
            ))
            prompt = f"(let [x {x_expr}] {body})"
        elif choice == "let-two":
            factor = self.literal(1, 5)
            x_expr = self.arithmetic(self.rng.randint(1, 2), low=0, high=9)
            y_expr = f"(* x {factor})"
            body = self.rng.choice((f"(- y x)", f"(+ y x)", f"(>= y x)"))
            prompt = f"(let [x {x_expr} y {y_expr}] {body})"
        elif choice == "compare":
            comparator = self.rng.choice(("=", "<", ">", "<=", ">="))
            prompt = f"({comparator} {self.arithmetic(2, low=0, high=8)} {self.arithmetic(2, low=0, high=8)})"
        else:
            left = self.literal(0, 9)
            right = self.literal(0, 9)
            prompt = self.rng.choice((
                f"(and (> {left} 0) (not (= {right} 99)))",
                f"(or (= {left} 0) (> {right} {left}))",
                f"(not (< {left} {right}))",
            ))
        return self.evaluated_example(2, choice, prompt)

    def stage_three(self) -> CorpusExample:
        choice = self.rng.choice(("quote", "eval", "list", "nested-list"))
        if choice == "quote":
            prompt = f"(quote {self.symbolic_source()})"
        elif choice == "eval":
            prompt = f"(eval (quote {self.arithmetic(self.rng.randint(1, 3), low=0, high=9)}))"
        elif choice == "list":
            prompt = f"(list {self.literal(0, 9)} {self.arithmetic(2, low=0, high=6)} :foo)"
        else:
            prompt = (
                f"(list (quote {self.symbolic_source()}) "
                f":value (eval (quote {self.arithmetic(2, low=0, high=6)})))"
            )
        return self.evaluated_example(3, choice, prompt)

    def stage_four(self) -> CorpusExample:
        choice = self.rng.choice(("if", "cond", "lambda", "define", "let-lambda"))
        value = self.literal(-5, 9)
        if choice == "if":
            prompt = (
                f"(if (> {value} 0) "
                f"{self.arithmetic(2, low=0, high=6)} "
                f"{self.arithmetic(2, low=0, high=6)})"
            )
        elif choice == "cond":
            prompt = f"(cond (= {value} 0) 0 (> {value} 0) (* {value} 2) :else (- 0 {value}))"
        elif choice == "lambda":
            body = self.rng.choice(("(+ x 2)", "(* x x)", "(- (* x 3) 1)", "(if (> x 0) x (- 0 x))"))
            prompt = f"((lambda [x] {body}) {value})"
        elif choice == "define":
            fn_name, body = self.rng.choice((
                ("double", "(* x 2)"),
                ("square", "(* x x)"),
                ("shift", "(+ x 3)"),
            ))
            prompt = f"(define {fn_name} (lambda [x] {body})) ({fn_name} {value})"
        else:
            delta = self.literal(1, 5)
            prompt = f"(let [f (lambda [x] (+ x {delta}))] (f {value}))"
        return self.evaluated_example(4, choice, prompt)

    def symbolic_source(self) -> str:
        x, y = self.rng.sample(("x", "y", "z"), 2)
        templates = (
            f"(+ {x} 0)",
            f"(* {x} 1)",
            f"(* {x} 0)",
            f"(+ {x} {self.literal(1, 5)})",
            f"(* {self.literal(2, 5)} {x})",
            f"(* {x} (+ {y} {self.literal(1, 5)}))",
            f"(expt {x} {self.literal(2, 3)})",
        )
        return self.rng.choice(templates)

    def stage_five(self) -> CorpusExample:
        choice = self.rng.choice(("simplify", "expand", "substitute", "differentiate"))
        variable = self.rng.choice(("x", "y", "z"))
        other = self.rng.choice([item for item in ("x", "y", "z") if item != variable])
        if choice == "simplify":
            expr = self.rng.choice((
                f"(+ {variable} 0)",
                f"(+ 0 {variable})",
                f"(* {variable} 1)",
                f"(* 1 {variable})",
                f"(* {variable} 0)",
                f"(+ (+ {variable} 0) {self.literal(1, 6)})",
                f"(* (* {self.literal(2, 6)} {variable}) 1)",
                f"(expt {variable} 1)",
                f"(expt {variable} 0)",
            ))
            prompt = f"(simplify (quote {expr}))"
        elif choice == "expand":
            expr = self.rng.choice((
                f"(* {variable} (+ {other} {self.literal(1, 5)}))",
                f"(* (+ {variable} {self.literal(1, 5)}) {other})",
                f"(* (+ {variable} {other}) (+ {variable} {self.literal(1, 5)}))",
            ))
            prompt = f"(expand (quote {expr}))"
        elif choice == "substitute":
            replacement = self.literal(-5, 5)
            expr = self.rng.choice((
                f"(+ {variable} 1)",
                f"(* {variable} {variable})",
                f"(* {self.literal(2, 5)} (+ {variable} {other}))",
                f"(+ (expt {variable} 2) {other})",
            ))
            prompt = f"(substitute (quote {expr}) (quote {variable}) {replacement})"
        else:
            expr = self.rng.choice((
                variable,
                f"(+ {variable} {self.literal(1, 5)})",
                f"(* {variable} {variable})",
                f"(* {self.literal(2, 5)} {variable})",
                f"(expt {variable} {self.literal(2, 4)})",
                f"(+ (expt {variable} 2) (* {self.literal(2, 5)} {variable}))",
            ))
            prompt = f"(differentiate (quote {expr}) (quote {variable}))"
        return self.evaluated_example(5, choice, prompt)

    def stage_six(self) -> CorpusExample:
        x, y, z = self.rng.sample(("x", "y", "z"), 3)
        cases = (
            (
                "simplify-add-zero",
                f"(+ {x} 0)",
                x,
                f"(step simplify-add-zero (+ {x} 0) {x})",
            ),
            (
                "simplify-mul-one",
                f"(* {x} 1)",
                x,
                f"(step simplify-mul-one (* {x} 1) {x})",
            ),
            (
                "simplify-mul-zero",
                f"(* {x} 0)",
                "0",
                f"(step simplify-mul-zero (* {x} 0) 0)",
            ),
            (
                "commute-add",
                f"(+ {x} {y})",
                f"(+ {y} {x})",
                f"(step commute-add (+ {x} {y}) (+ {y} {x}))",
            ),
            (
                "commute-mul",
                f"(* {x} {y})",
                f"(* {y} {x})",
                f"(step commute-mul (* {x} {y}) (* {y} {x}))",
            ),
            (
                "distribute",
                f"(* {x} (+ {y} {z}))",
                f"(+ (* {x} {y}) (* {x} {z}))",
                f"(step distribute (* {x} (+ {y} {z})) (+ (* {x} {y}) (* {x} {z})))",
            ),
            (
                "simplify-add-zero-left",
                f"(+ 0 {x})",
                x,
                f"(step simplify-add-zero (+ 0 {x}) {x})",
            ),
            (
                "simplify-mul-one-left",
                f"(* 1 {x})",
                x,
                f"(step simplify-mul-one (* 1 {x}) {x})",
            ),
        )
        kind, premise, target, proof = self.rng.choice(cases)
        target_expr = parse_program(target)[0]
        premise_expr = parse_program(premise)[0]
        validation = validate_proof(proof, target=target_expr, premises=(premise_expr,))
        if not validation.complete:
            raise RuntimeError(f"Generated invalid proof: {proof}")
        prompt = f"(prove {target} from {premise})"
        return CorpusExample(stage=6, kind=kind, prompt=prompt, answer=proof)

    def vector_literal(self, size: int | None = None) -> str:
        size = size or self.rng.randint(2, 5)
        return "[" + " ".join(str(self.literal(0, 9)) for _ in range(size)) + "]"

    def stage_seven(self) -> CorpusExample:
        vector = self.vector_literal()
        choice = self.rng.choice((
            "get-vector",
            "count-vector",
            "conj-vector",
            "assoc-vector",
            "get-map",
            "assoc-map",
            "mapv-lambda",
            "mapv-partial",
            "filterv",
            "reduce",
            "thread-first",
            "thread-last",
            "comp",
        ))
        if choice == "get-vector":
            size = len(vector.strip("[]").split())
            prompt = f"(get {vector} {self.rng.randrange(size)})"
        elif choice == "count-vector":
            prompt = f"(count {vector})"
        elif choice == "conj-vector":
            prompt = f"(conj {vector} {self.literal(0, 9)})"
        elif choice == "assoc-vector":
            size = len(vector.strip("[]").split())
            prompt = f"(assoc {vector} {self.rng.randrange(size)} {self.literal(0, 9)})"
        elif choice == "get-map":
            prompt = self.rng.choice(("(get {:a 1 :b 2 :c 3} :a)", "(get {:a 1 :b 2 :c 3} :z 0)"))
        elif choice == "assoc-map":
            prompt = f"(assoc {{:a 1 :b 2}} :c {self.literal(0, 9)})"
        elif choice == "mapv-lambda":
            prompt = f"(mapv (lambda [x] (* x {self.literal(2, 4)})) {vector})"
        elif choice == "mapv-partial":
            prompt = f"(mapv (partial + {self.literal(1, 5)}) {vector})"
        elif choice == "filterv":
            prompt = f"(filterv (lambda [x] (> x {self.literal(2, 6)})) {vector})"
        elif choice == "reduce":
            prompt = f"(reduce + 0 {vector})"
        elif choice == "thread-first":
            prompt = f"(-> {vector} (conj {self.literal(0, 9)}) count)"
        elif choice == "thread-last":
            prompt = f"(->> {vector} (mapv (partial + {self.literal(1, 5)})) (reduce + 0))"
        else:
            prompt = f"((comp (partial * {self.literal(2, 4)}) (partial + {self.literal(1, 5)})) {self.literal(0, 9)})"
        return self.evaluated_example(7, choice, prompt)


def write_readme(output_dir: Path, total_chars: int, total_examples: int, seed: int) -> None:
    readme = f"""# Lisp Math Corpus

Deterministic synthetic pretraining corpus for the repository's custom Lisp
mathematics dialect.

- Generator: `scripts/generate_lisp_math_corpus.py`
- Seed: `{seed}`
- Examples: `{total_examples:,}`
- Characters: `{total_chars:,}`
- Format: prompt line, answer/proof line(s), blank line
- Dialect source of truth: `src/reciprocator/rl/lisp_eval.py`

The corpus follows the RL mathematical curriculum:

1. Arithmetic: `+`, `-`, `*`, `/`, `mod`, `expt`
2. Local bindings, comparisons, booleans: `let`, `=`, `<`, `>`, `<=`, `>=`, `and`, `or`, `not`
3. Code as data: `quote`, `list`, `eval`
4. Conditionals and functions: `if`, `cond`, `define`, `lambda`
5. Symbolic algebra: `simplify`, `expand`, `substitute`, `differentiate`
6. Proof construction: validated `(step rule input output)` sequences
7. Collections and higher-order operations: vectors, maps, `get`, `count`, `conj`, `assoc`, `mapv`, `filterv`, `reduce`, `->`, `->>`, `comp`, `partial`

Example:

```lisp
(+ 1 (* 2 3))
7

(prove x from (+ x 0))
(step simplify-add-zero (+ x 0) x)
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_sources(output_dir: Path, counts: dict[tuple[int, str], list[int]]) -> None:
    lines = ["stage\tkind\texamples\tcharacters\n"]
    for (stage, kind), (examples, chars) in sorted(counts.items()):
        lines.append(f"{stage}\t{kind}\t{examples}\t{chars}\n")
    (output_dir / "sources.tsv").write_text("".join(lines), encoding="utf-8")


def generate_corpus(output_dir: Path, target_chars: int, seed: int) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / COMBINED_FILENAME
    tmp_path = output_dir / f".{COMBINED_FILENAME}.tmp"
    rng = random.Random(seed)
    generator = LispMathExampleGenerator(rng)
    counts: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    total_chars = 0
    total_examples = 0

    with tmp_path.open("w", encoding="utf-8") as handle:
        while total_chars < target_chars:
            example = generator.generate()
            block = example.block
            handle.write(block)
            block_chars = len(block)
            total_chars += block_chars
            total_examples += 1
            stats = counts[(example.stage, example.kind)]
            stats[0] += 1
            stats[1] += block_chars

    tmp_path.replace(combined_path)
    write_sources(output_dir, counts)
    write_readme(output_dir, total_chars, total_examples, seed)
    return total_chars, total_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "corpora" / CORPUS_NAME,
        help="Directory to write README.md, sources.tsv, and the combined corpus file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_chars <= 0:
        raise SystemExit("--target-chars must be positive.")
    total_chars, total_examples = generate_corpus(args.output_dir, args.target_chars, args.seed)
    print(
        f"Wrote {total_examples:,} examples / {total_chars:,} chars "
        f"to {args.output_dir / COMBINED_FILENAME}"
    )


if __name__ == "__main__":
    main()
