import sys

import torch

sys.path.insert(0, ".")
from kvstamp.compress import alpha_rescale, compress_kv, expand_kv, reconstruction_error, score_tokens  # noqa: E402

torch.manual_seed(0)


def _mk(t=128, h=4, d=64, tq=8):
    q = torch.randn(h, tq, d)
    k = torch.randn(t, d)
    v = torch.randn(t, d)
    # heavy tokens: 6 positions get keys nearly parallel to a real query
    heavy_idx = torch.randperm(t)[:6]
    for i in heavy_idx:
        k[i] = 5.0 * q[0, int(i) % tq] + torch.randn(d) * 0.01
    return q, k, v, heavy_idx


def test_scores_concentrate_on_heavy_tokens():
    q, k, v, heavy_idx = _mk()
    s = score_tokens(q, k)
    top = set(torch.argsort(s, descending=True)[:6].tolist())
    overlap = len(top & {int(i) for i in heavy_idx})
    assert overlap >= 4, (top, heavy_idx.tolist())


# the other tests construct their own tensors
def test_compress_keeps_sinks_and_budget():
    q, k, v, _ = _mk(t=128)
    s = score_tokens(q, k)
    packed = compress_kv(k, v, s, budget=32)
    assert packed["keys"].shape[0] == 32
    assert packed["keep_idx"][:4].tolist() == [0, 1, 2, 3]  # sinks preserved
    assert packed["kept_fraction"] == 32 / 128


def test_expand_roundtrip_positions():
    q, k, v, _ = _mk(t=64)
    s = score_tokens(q, k)
    packed = compress_kv(k, v, s, budget=16)
    ek, ev = expand_kv(packed, 64)
    assert ek.shape == k.shape
    assert torch.equal(ek[packed["keep_idx"]], packed["keys"])
    dropped = ~torch.zeros(64, dtype=torch.bool).index_fill(0, packed["keep_idx"], True)
    assert (ek[dropped] == 0).all()


def test_reconstruction_error_small_when_heavy_kept():
    q, k, v, _ = _mk(t=256)
    s = score_tokens(q, k)
    low = reconstruction_error(q, k, v, s, budget=16)   # drops most heavy tokens
    high = reconstruction_error(q, k, v, s, budget=64)  # keeps more
    assert high["relative_output_error"] < low["relative_output_error"]


def test_budget_full_gives_zero_error():
    q, k, v, _ = _mk(t=64)
    s = score_tokens(q, k)
    full = reconstruction_error(q, k, v, s, budget=64)
    assert full["relative_output_error"] < 1e-6


def test_alpha_rescale_renormalizes():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    keep = torch.tensor([True, False, True, False])
    rescaled = alpha_rescale(logits, keep)
    p_before = torch.softmax(logits, -1)[..., keep].sum()
    p_after = torch.softmax(rescaled, -1)[..., keep].sum()
    assert p_after > p_before, "rescale must move mass onto kept tokens"
    # dropped positions must not contribute after rescale
    assert torch.isinf(rescaled[..., ~keep]).all()
