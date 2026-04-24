import copy

import torch
import torch.nn.functional as F

from reciprocator.rl.curriculum import CurriculumController
from reciprocator.model import ReciprocatorLM
from reciprocator.rl.problem_gen import DifficultyConfig, ProblemExample
from reciprocator.rl.training import LispGRPOConfig, _sequence_statistics, build_lisp_tokenizer, train_lisp_grpo


class FixedAdditionGenerator:
    def __init__(self) -> None:
        self.problem = ProblemExample(
            prompt_expression="(+ 1 2)",
            expected_result=3,
            difficulty=DifficultyConfig(stage=1, depth=1, operator_set=("+",), value_range=(0, 4)),
        )

    def generate_batch(self, difficulties):
        return [self.problem for _ in difficulties]


def _incremental_sequence_statistics(model, reference_model, tokenizer, prompt, completion_ids, *, device):
    prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
    logits, states = model(prompt_tokens, position_offset=0)
    with torch.no_grad():
        ref_logits, ref_states = reference_model(prompt_tokens, position_offset=0)
    current_logits = logits[:, -1]
    current_ref_logits = ref_logits[:, -1]
    log_prob_terms = []
    kl_terms = []
    position = int(prompt_tokens.shape[1])

    for index, token_id in enumerate(completion_ids):
        log_probs = F.log_softmax(current_logits, dim=-1)
        probabilities = log_probs.exp()
        with torch.no_grad():
            ref_log_probs = F.log_softmax(current_ref_logits, dim=-1)
        log_prob_terms.append(log_probs[0, token_id])
        kl_terms.append(torch.sum(probabilities * (log_probs - ref_log_probs), dim=-1).squeeze(0))
        if index == len(completion_ids) - 1:
            break
        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
        logits, states = model(token_tensor, states=states, position_offset=position)
        current_logits = logits[:, -1]
        with torch.no_grad():
            ref_logits, ref_states = reference_model(token_tensor, states=ref_states, position_offset=position)
            current_ref_logits = ref_logits[:, -1]
        position += 1
    return torch.stack(log_prob_terms).mean(), torch.stack(kl_terms).mean()


def test_sequence_statistics_matches_incremental_teacher_forcing() -> None:
    torch.manual_seed(0)
    tokenizer = build_lisp_tokenizer()
    model = ReciprocatorLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=12,
        state_shape=(2, 2),
        num_layers=1,
        token_frequencies=torch.ones(tokenizer.vocab_size),
    )
    reference_model = copy.deepcopy(model).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    completion_ids = tuple(int(token) for token in tokenizer.encode("12)"))

    optimized_log_prob, optimized_kl = _sequence_statistics(
        model,
        reference_model,
        tokenizer,
        "(+ 1 2)\n",
        completion_ids,
        device=torch.device("cpu"),
    )
    incremental_log_prob, incremental_kl = _incremental_sequence_statistics(
        model,
        reference_model,
        tokenizer,
        "(+ 1 2)\n",
        completion_ids,
        device=torch.device("cpu"),
    )

    assert torch.allclose(optimized_log_prob, incremental_log_prob, atol=1e-6)
    assert torch.allclose(optimized_kl, incremental_kl, atol=1e-6)


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
    assert metrics.grad_norm >= 0.0
    assert metrics.mean_phase_variance >= 0.0
    assert metrics.samples[0].prompt_expression == "(+ 1 2)"
    assert metrics.samples[0].completion == "3"
