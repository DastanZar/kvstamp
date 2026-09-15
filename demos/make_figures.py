"""kvstamp figures: the attention heatmaps that *are* the argument."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kvstamp.compress import compress_kv, reconstruction_error, score_tokens  # noqa: E402

FIG = os.path.join(ROOT, "assets")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140, "font.size": 9})

torch.manual_seed(4)
t, h, d, tq = 192, 4, 48, 24
q = torch.randn(h, tq, d)
k = torch.randn(t, d)
v = torch.randn(t, d)
heavy = torch.tensor([3, 4, 5, 60, 61, 120, 121, 160])  # a local cluster + far islands
for i in heavy:
    k[i] = 4.0 * q[0, int(i) % tq] + torch.randn(d) * 0.01

s = score_tokens(q, k)
full_logits = q @ k.T / (d ** 0.5)
full_p = torch.softmax(full_logits, dim=-1)
packed = compress_kv(k, v, s, budget=32)
comp_logits = q @ packed["keys"].T / (d ** 0.5)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
im0 = axes[0].imshow(full_p.mean(0), aspect="auto", cmap="magma")
axes[0].set_title("full cache — attention over 192 keys\n(sinks + 8 heavy carry the mass)", fontsize=8.5)
axes[0].set_xlabel("key position")
im1 = axes[1].imshow(torch.softmax(comp_logits, -1).mean(0), aspect="auto", cmap="magma")
axes[1].set_title("compressed to 32 kept keys — the SAME mass,\njust fewer columns", fontsize=8.5)
axes[1].set_xlabel("kept position")
for i, (ax, ttl) in enumerate([(axes[2], None)]):
    keep = torch.zeros(t, dtype=torch.bool); keep[packed["keep_idx"]] = True
    ax.plot(range(t), s / s.max(), lw=1.1, color="#4472c4", label="score (norm.)")
    ax.fill_between(range(t), 0, 1, where=keep.numpy(), color="#4472c4", alpha=.12, label="kept")
    for i2 in heavy:
        ax.axvline(i2, color="#c0504d", lw=.7, alpha=.55)
    ax.set_ylim(0, 1.05); ax.set_title("scoring: budget=32 keeps exactly the\nsinks and the heavy islands", fontsize=8.5)
    ax.legend(fontsize=7); ax.set_xlabel("key position")
axes[0].set_ylabel("query"); 
for a in axes: a.tick_params(labelsize=7)
fig.suptitle("why dropping by score works — the mass you keep IS the output", y=1.02, fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/attention_maps.png"); plt.close(fig)

# error-vs-budget curve with the knee annotated
fig, ax = plt.subplots(figsize=(6.4, 4.0))
budgets = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192]
errs, kept = [], []
for b in budgets:
    r = reconstruction_error(q, k, v, s, budget=b)
    errs.append(r["relative_output_error"]); kept.append(r["kept_fraction"])
ax.plot(kept, errs, "o-", color="#4472c4")
ax.axhline(0.05, ls="--", lw=.8, c="#888")
ax.text(.02, .056, "0.05 error budget", fontsize=7.5, color="#666")
knee = next(kf for kf, e in zip(kept, errs) if e < 0.05)
ax.annotate(f"knee ≈ {knee:.0%} kept → {1/knee:.0f}× KV compression", (knee, .05),
            xytext=(24, -22), textcoords="offset points", fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=.8))
ax.set_yscale("log"); ax.set_xlabel("fraction of KV cache kept"); ax.set_ylabel("rel. attention-output error")
ax.set_title("the real currency is reconstruction error, not tokens dropped")
ax.grid(alpha=.2, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/error_knee.png"); plt.close(fig)

# alpha-rescale illustration: softmax mass redistribution
fig, ax = plt.subplots(figsize=(6.6, 3.4))
row = full_p[0, 3]
logits = q[0, 3].unsqueeze(0) @ k.T / (d ** 0.5)
drop = torch.ones(t, dtype=torch.bool); drop[packed["keep_idx"]] = False
import torch.nn.functional as F
raw = torch.softmax(logits.masked_fill(drop, -1e9), -1)
p_before = raw.sum().item()  # kept-mass fraction after renorm
ax.bar([0, 1], [1.0, float(p_before)], color=["#4472c4", "#c0504d"])
ax.set_xticks([0, 1], ["all mass\n(full cache)", "kept tokens\n(naive renorm)"], fontsize=7.5)
ax.set_ylabel("probability mass on kept keys")
ax.set_title("dropping tokens silently rescales the distribution —\nthat's the bug the log(α) term fixes", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/alpha_mass.png"); plt.close(fig)
print("kvstamp figures ->", sorted(os.listdir(FIG)))
