#!/usr/bin/env python3
"""Train a real-valued Mamba-style selective SSM character-LM baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reciprocator.training import build_corpus_dataset, sample_causal_lm_batch  # noqa: E402


@dataclass(frozen=True)
class MambaConfig:
    corpus_name: str
    max_chars: Optional[int]
    val_fraction: float
    batch_size: int
    seq_len: int
    steps: int
    eval_every: int
    eval_batches: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    num_layers: int
    expansion_factor: int
    state_size: int
    conv_kernel: int
    dt_rank: int
    dropout: float
    device: str
    seed: int
    run_name: str
    checkpoint_dir: str
    checkpoint_every: int
    taper_patience: int
    taper_min_delta: float
    min_steps_before_taper: int


class MambaBlock(nn.Module):
    """Small diagonal selective SSM block inspired by Mamba.

    This favors clarity and local availability over the official fused selective-scan
    kernel. It is suitable as a baseline, but not as an optimized Mamba implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        expansion_factor: int,
        state_size: int,
        conv_kernel: int,
        dt_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.inner_size = hidden_size * expansion_factor
        self.state_size = state_size
        self.dt_rank = dt_rank
        self.norm = nn.LayerNorm(hidden_size)
        self.in_proj = nn.Linear(hidden_size, 2 * self.inner_size)
        self.conv = nn.Conv1d(
            self.inner_size,
            self.inner_size,
            kernel_size=conv_kernel,
            groups=self.inner_size,
            padding=conv_kernel - 1,
        )
        self.x_proj = nn.Linear(self.inner_size, dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.inner_size)
        a = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.inner_size, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.inner_size))
        self.out_proj = nn.Linear(self.inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        x, gate = self.in_proj(hidden).chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv(x)[..., : hidden.shape[1]].transpose(1, 2)
        x = F.silu(x)

        params = self.x_proj(x)
        dt_raw, b_raw, c_raw = torch.split(params, [self.dt_rank, self.state_size, self.state_size], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        b = b_raw
        c = c_raw
        a = -torch.exp(self.A_log)

        state = x.new_zeros(x.shape[0], self.inner_size, self.state_size)
        outputs: list[Tensor] = []
        for t in range(x.shape[1]):
            dt_t = dt[:, t, :].unsqueeze(-1)
            x_t = x[:, t, :].unsqueeze(-1)
            decay = torch.exp(dt_t * a.unsqueeze(0))
            state = decay * state + dt_t * b[:, t, :].unsqueeze(1) * x_t
            y_t = (state * c[:, t, :].unsqueeze(1)).sum(dim=-1) + self.D.unsqueeze(0) * x[:, t, :]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(gate)
        return residual + self.dropout(self.out_proj(y))


class MambaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        expansion_factor: int,
        state_size: int,
        conv_kernel: int,
        dt_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    hidden_size,
                    expansion_factor=expansion_factor,
                    state_size=state_size,
                    conv_kernel=conv_kernel,
                    dt_rank=dt_rank,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


def metrics_from_loss_and_logits(loss: Tensor, logits: Tensor, targets: Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).to(torch.float32).mean().item()
    loss_value = float(loss.item())
    return {
        "loss": loss_value,
        "accuracy": accuracy,
        "perplexity": float(math.exp(min(loss_value, 20.0))),
        "bpc": loss_value / math.log(2.0),
    }


def evaluate(model: MambaLM, tokens: Tensor, *, batch_size: int, seq_len: int, batches: int, device: torch.device) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = sample_causal_lm_batch(tokens, batch_size, seq_len, device=device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            rows.append(metrics_from_loss_and_logits(loss, logits, targets))
    model.train()
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="greek_classics")
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--expansion-factor", type=int, default=2)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--dt-rank", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-dir", default="runs")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--taper-patience", type=int, default=10)
    parser.add_argument("--taper-min-delta", type=float, default=0.01)
    parser.add_argument("--min-steps-before-taper", type=int, default=500)
    return parser.parse_args(argv)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, *, model: MambaLM, optimizer: torch.optim.Optimizer, config: MambaConfig, step: int, metrics: list[dict[str, object]], dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
            "vocabulary": list(dataset.tokenizer.itos),
            "train_token_count": int(dataset.train_tokens.numel()),
            "val_token_count": int(dataset.val_tokens.numel()),
        },
        path,
    )
    print(f"saved checkpoint: {path.resolve()}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.max_chars is not None and args.max_chars <= 0:
        args.max_chars = None
    config = MambaConfig(
        corpus_name=args.corpus,
        max_chars=args.max_chars,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        expansion_factor=args.expansion_factor,
        state_size=args.state_size,
        conv_kernel=args.conv_kernel,
        dt_rank=args.dt_rank,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        taper_patience=args.taper_patience,
        taper_min_delta=args.taper_min_delta,
        min_steps_before_taper=args.min_steps_before_taper,
    )
    torch.manual_seed(config.seed)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = build_corpus_dataset(config.corpus_name, max_chars=config.max_chars, val_fraction=config.val_fraction)
    model = MambaLM(
        vocab_size=dataset.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        expansion_factor=config.expansion_factor,
        state_size=config.state_size,
        conv_kernel=config.conv_kernel,
        dt_rank=config.dt_rank,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    run_dir = (REPO_ROOT / config.checkpoint_dir / config.run_name).resolve()
    total_params = sum(p.numel() for p in model.parameters())
    write_json(
        run_dir / "config.json",
        {
            **asdict(config),
            "vocab_size": dataset.vocab_size,
            "train_token_count": int(dataset.train_tokens.numel()),
            "val_token_count": int(dataset.val_tokens.numel()),
            "total_params": total_params,
        },
    )
    print(
        f"mamba config: params={total_params} vocab={dataset.vocab_size} "
        f"train_tokens={dataset.train_tokens.numel()} val_tokens={dataset.val_tokens.numel()}",
        flush=True,
    )

    metrics: list[dict[str, object]] = []
    best_val_bpc = float("inf")
    stale_evals = 0
    final_step = 0
    for step in range(1, config.steps + 1):
        final_step = step
        inputs, targets = sample_causal_lm_batch(dataset.train_tokens, config.batch_size, config.seq_len, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            train_metrics = metrics_from_loss_and_logits(loss.detach(), logits.detach(), targets)
            val_metrics = evaluate(
                model,
                dataset.val_tokens,
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                batches=config.eval_batches,
                device=device,
            )
            row = {"step": step, "train": train_metrics, "val": val_metrics}
            metrics.append(row)
            write_json(run_dir / "metrics.json", metrics)
            print(
                f"step={step} lr={config.learning_rate:.6g} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"train_bpc={train_metrics['bpc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_bpc={val_metrics['bpc']:.4f}",
                flush=True,
            )
            if val_metrics["bpc"] < best_val_bpc - config.taper_min_delta:
                best_val_bpc = val_metrics["bpc"]
                stale_evals = 0
            else:
                stale_evals += 1
            if (
                step >= config.min_steps_before_taper
                and config.taper_patience > 0
                and stale_evals >= config.taper_patience
            ):
                print(
                    f"taper stop: no val_bpc improvement > {config.taper_min_delta} "
                    f"for {stale_evals} evals; best_val_bpc={best_val_bpc:.4f}",
                    flush=True,
                )
                break

        if config.checkpoint_every > 0 and step % config.checkpoint_every == 0:
            save_checkpoint(
                run_dir / f"checkpoint_step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                metrics=metrics,
                dataset=dataset,
            )

    save_checkpoint(
        run_dir / f"checkpoint_step_{final_step:06d}.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=final_step,
        metrics=metrics,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
