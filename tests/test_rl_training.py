import torch

from reciprocator.rl.curriculum import CurriculumController
from reciprocator.model import ReciprocatorLM
from reciprocator.rl.problem_gen import DifficultyConfig, ProblemExample
from reciprocator.rl.training import LispGRPOConfig, build_lisp_tokenizer, train_lisp_grpo


class FixedAdditionGenerator:
    def __init__(self) -> None:
        self.problem = ProblemExample(
            prompt_expression="(+ 1 2)",
            expected_result=3,
            difficulty=DifficultyConfig(stage=1, depth=1, operator_set=("+",), value_range=(0, 4)),
        )

    def generate_batch(self, difficulties):
        return [self.problem for _ in difficulties]


def test_train_lisp_grpo_runs_fixed_addition_end_to_end() -> None:
    tokenizer = build_lisp_tokenizer()
    config = LispGRPOConfig(
        steps=1,
        batch_size=2,
        group_size=2,
        max_completion_tokens=1,
        temperature=0.0,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        device="cpu",
        sample_record_limit=2,
    )
    model = ReciprocatorLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        state_shape=config.state_shape,
        num_layers=config.num_layers,
        token_frequencies=torch.ones(tokenizer.vocab_size),
    )

    with torch.no_grad():
        model.readout.output.weight.zero_()
        model.readout.output.bias.zero_()
        model.readout.output.bias[tokenizer.stoi["3"]] = 10.0

    result = train_lisp_grpo(
        config,
        tokenizer=tokenizer,
        model=model,
        curriculum=CurriculumController(current_stage=1, harder_stage_mix=0.0),
        generator=FixedAdditionGenerator(),
    )

    metrics = result.step_metrics[0]
    assert metrics.mean_reward == 1.0
    assert metrics.mean_phase_variance >= 0.0
    assert metrics.samples[0].prompt_expression == "(+ 1 2)"
    assert metrics.samples[0].completion == "3"
