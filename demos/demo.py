"""Demo: budget sweep — reconstruction error vs KV bytes saved."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvstamp.compress import reconstruction_error, score_tokens  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "receipts")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(7)
t, h, d, tq = 512, 8, 64, 16
q = torch.randn(h, tq, d)
k = torch.randn(t, d)
v = torch.randn(t, d)
for i in torch.randperm(t)[:10]:
    k[i] = q[0, i % tq].clone() + torch.randn(d) * 0.1  # heavy tokens

s = score_tokens(q, k)
sweep = {}
for budget in (32, 64, 128, 256, 512):
    r = reconstruction_error(q, k, v, s, budget=budget)
    sweep[f"budget={budget}/{t}"] = {
        "kept_fraction": round(r["kept_fraction"], 3),
        "relative_output_error": round(r["relative_output_error"], 4),
    }

receipt = {
    "context_tokens": t,
    "sweep": sweep,
    "kv_bytes_full_fp16": t * 2 * d * 2,
    "kv_bytes_at_64": 64 * 2 * d * 2,
    "compression_at_64": f"{t / 64:.1f}x",
}
with open(os.path.join(OUT, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = [32, 64, 128, 256, 512]
    errs = [sweep[f"budget={b}/{t}"]["relative_output_error"] for b in budgets]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(budgets, errs, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("kept KV tokens (of 512)")
    ax.set_ylabel("relative attention-output error")
    ax.set_title("KV compression: budget vs reconstruction error")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kvstamp.png"), dpi=140)
    print("chart -> receipts/kvstamp.png")
except Exception as e:  # pragma: no cover
    print("chart skipped:", e)

assert sweep["budget=512/512"]["relative_output_error"] < 1e-6
assert sweep["budget=64/512"]["relative_output_error"] < sweep["budget=32/512"]["relative_output_error"]
print("DEMO_OK")
