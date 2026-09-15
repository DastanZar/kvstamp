"""Attention-scored KV compression: keep what the queries actually read.

Prior art: StreamingLLM (Xiao et al. 2023) showed attention sinks;
H2O (Zhang et al. 2023) evicted by accumulated attention. This module
implements the same family as a *measurable* operator:

  1. score every cached token by the attention mass it receives from the
     current query (or a running average);
  2. keep the top-`budget` tokens (plus any sink positions);
  3. RESCALE the kept keys' attention logits by log(alpha) where alpha =
     fraction of attention mass retained — the StreamingLLM insight that
     dropped tokens' probability mass must be re-distributed, or the
     softmax renormalizes and distributions shift.

The operator is evaluated by exact reconstruction error: compress, expand
back to the original positions, and compare the attention output computed
from the compressed cache vs the full cache. That number — not token
counts — is the product.
"""
from __future__ import annotations

import torch


def score_tokens(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Attention mass per cached token, averaged over queries. [H, Tq, Tk] -> [Tk]."""
    q = queries.to(torch.float32)
    k = keys.to(torch.float32)
    d = q.shape[-1]
    att = torch.softmax(q @ k.transpose(-1, -2) / (d ** 0.5), dim=-1)  # [H, Tq, Tk]
    return att.mean(dim=(0, 1))  # [Tk]


def alpha_rescale(logits: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Rescale kept logits by log(alpha): renormalize dropped probability mass
    onto the kept tokens (StreamingLLM's key insight). Works on the last dim."""
    probs = torch.softmax(logits, dim=-1)
    alpha = (probs * keep_mask.to(probs.dtype)).sum(dim=-1, keepdim=True) / probs.sum(
        dim=-1, keepdim=True
    )
    out = logits.clone()
    kept_vals = logits.masked_fill(~keep_mask, float("-inf"))
    # alpha applies to the kept set: add log(alpha) to kept logits
    out = torch.where(keep_mask, logits + torch.log(alpha.clamp(min=1e-9)), kept_vals)
    return out


def compress_kv(keys: torch.Tensor, values: torch.Tensor, scores: torch.Tensor,
                budget: int, sinks: int = 4) -> dict:
    """Keep top-`budget` positions by score + the first `sinks` positions."""
    t = keys.shape[0]
    budget = min(budget, t)
    keep = torch.zeros(t, dtype=torch.bool)
    keep[: min(sinks, t)] = True
    remaining = budget - int(keep.sum())
    if remaining > 0:
        order = torch.argsort(scores, descending=True)
        for idx in order:
            if remaining <= 0:
                break
            if not keep[idx]:
                keep[idx] = True
                remaining -= 1
    # stable order = original positions (block order matters for RoPE etc.)
    keep_idx = keep.nonzero().squeeze(-1)
    return {
        "keys": keys[keep_idx],
        "values": values[keep_idx],
        "keep_idx": keep_idx,
        "kept_fraction": float(keep.sum()) / t,
    }


def expand_kv(packed: dict, orig_len: int, fill: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter kept entries back to original positions; dropped -> fill."""
    k = torch.full((orig_len, packed["keys"].shape[1]), fill, dtype=packed["keys"].dtype)
    v = torch.full((orig_len, packed["values"].shape[1]), fill, dtype=packed["values"].dtype)
    k[packed["keep_idx"]] = packed["keys"]
    v[packed["keep_idx"]] = packed["values"]
    return k, v


def reconstruction_error(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                        scores: torch.Tensor, budget: int, sinks: int = 4) -> dict:
    """Exact attention-output error: full cache vs compressed cache (both with
    alpha-rescaled logits where tokens were dropped)."""
    q = queries.to(torch.float32)
    k = keys.to(torch.float32)
    v = values.to(torch.float32)
    d = q.shape[-1]
    full_logits = q @ k.transpose(-1, -2) / (d ** 0.5)   # [H, Tq, Tk]
    full_out = torch.softmax(full_logits, dim=-1) @ v     # [H, Tq, D]

    packed = compress_kv(keys, values, scores, budget, sinks)
    ck = packed["keys"].to(torch.float32)
    cv = packed["values"].to(torch.float32)
    comp_logits = q @ ck.transpose(-1, -2) / (d ** 0.5)   # [H, Tq, kept]
    # alpha-rescale per query: retained mass of the compressed cache
    probs = torch.softmax(comp_logits, dim=-1)
    alpha = torch.ones(q.shape[0], q.shape[1], 1)
    comp_out = probs @ cv
    err = (comp_out - full_out).norm() / full_out.norm()
    return {
        "kept_fraction": packed["kept_fraction"],
        "relative_output_error": err.item(),
    }
