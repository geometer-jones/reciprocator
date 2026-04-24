#!/usr/bin/env python3
"""Train a real-valued causal Transformer character-LM baseline."""

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
class TransformerConfig:
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
    num_heads: int
    ffn_expansion_factor: int
    dropout: float
    device: str
    seed: int
    run_name: str
    checkpoint_dir: str
    checkpoint_every: int
    taper_patience: int
    taper_min_delta: float
    min_steps_before_taper: int


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq.to(device))
        return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [B, H, T, Dh], cos/sin: [T, Dh/2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope(seq_len, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion_factor: int) -> None:
        super().__init__()
        inner = hidden_size * expansion_factor
        self.gate = nn.Linear(hidden_size, inner)
        self.up = nn.Linear(hidden_size, inner)
        self.down = nn.Linear(inner, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_expansion_factor: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, ffn_expansion_factor)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, ffn_expansion_factor, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.output(self.final_norm(x))

    def loss(self, input_ids: Tensor, targets: Tensor) -> Tensor:
        logits = self(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


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


def evaluate(model: TransformerLM, tokens: Tensor, *, batch_size: int, seq_len: int, batches: int, device: torch.device) -> dict[str, float]:
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
    parser.add_argument("--hidden-size", type=int, default=480)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-expansion-factor", type=int, default=4)
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


def save_checkpoint(path: Path, *, model: TransformerLM, optimizer: torch.optim.Optimizer, config: TransformerConfig, step: int, metrics: list[dict[str, object]], dataset) -> None:
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
    config = TransformerConfig(
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
        num_heads=args.num_heads,
        ffn_expansion_factor=args.ffn_expansion_factor,
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
    model = TransformerLM(
        vocab_size=dataset.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_expansion_factor=config.ffn_expansion_factor,
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
        f"transformer config: params={total_params} vocab={dataset.vocab_size} "
        f"train_tokens={dataset.train_tokens.numel()} val_tokens={dataset.val_tokens.numel()}",
        flush=True,
    )

    metrics: list[dict[str, object]] = []
    best_val_bpc = float("inf")
    stale_evals = 0
    for step in range(1, config.steps + 1):
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
        run_dir / f"checkpoint_step_{step:06d}.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=step,
        metrics=metrics,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
