"""Re-rotation identity: rope(k, p) rotated by -delta == rope(k, p - delta)."""

import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from sinkcache.rope import apply_rope, rerotate, rope_cos_sin

from .conftest import DEVICE

HEAD_DIM = 128
THETA = 1_000_000.0  # Qwen3


def inv_freq(dim=HEAD_DIM, theta=THETA):
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def test_apply_rope_matches_hf():
    """Our rotation is the same op as HF's apply_rotary_pos_emb."""
    torch.manual_seed(0)
    k = torch.randn(2, 8, 37, HEAD_DIM)
    pos = torch.arange(1000, 1037)
    cos, sin = rope_cos_sin(pos, inv_freq())
    _, k_hf = apply_rotary_pos_emb(k, k, cos[None], sin[None])
    torch.testing.assert_close(apply_rope(k, pos, inv_freq()), k_hf, rtol=0, atol=0)


@pytest.mark.parametrize("p", [5, 1024, 40_000])
@pytest.mark.parametrize("delta", [1, 7, 1000])
def test_rerotate_fp32(p, delta):
    torch.manual_seed(p + delta)
    k = torch.randn(1, 8, 16, HEAD_DIM)
    pos = torch.arange(p, p + 16) + delta
    shifted = rerotate(apply_rope(k, pos, inv_freq()), delta, inv_freq())
    direct = apply_rope(k, pos - delta, inv_freq())
    # The fp32 angle p*inv_freq[0] = p radians carries ~p * 2^-24 of rounding, so the
    # achievable tolerance grows with position (as it does inside HF itself).
    atol = 4e-7 * (p + delta + 16) + 1e-6
    torch.testing.assert_close(shifted, direct, rtol=0, atol=atol)


def test_rerotate_bf16_one_rounding():
    """In bf16 the re-rotated key differs from the direct one by about one rounding (~2^-8 relative)."""
    torch.manual_seed(0)
    k = torch.randn(1, 8, 64, HEAD_DIM)
    pos = torch.arange(500, 564)
    stored = apply_rope(k, pos + 3, inv_freq()).bfloat16()
    shifted = rerotate(stored, 3, inv_freq()).float()
    direct = apply_rope(k, pos, inv_freq())
    rel = (shifted - direct).norm() / direct.norm()
    assert rel < 2 ** -7, rel


def test_rotation_order_matters():
    """Negative control: if a per-channel norm were applied *after* RoPE, re-rotation would not hold.
    Qwen3 applies k_norm *before* RoPE, which is why it is safe (see test_qwen3_keys_shift below)."""
    torch.manual_seed(0)
    k = torch.randn(1, 1, 4, HEAD_DIM)
    w = torch.rand(HEAD_DIM) + 0.5  # non-uniform per-channel weight, like RMSNorm's
    norm = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    pos = torch.arange(100, 104)
    shifted = rerotate(norm(apply_rope(k, pos, inv_freq())), 10, inv_freq())
    direct = norm(apply_rope(k, pos - 10, inv_freq()))
    assert not torch.allclose(shifted, direct, atol=1e-2)


@torch.no_grad()
def test_qwen3_keys_shift(model_fp32, text_ids):
    """Real Qwen3 (k_norm -> RoPE): run the same tokens at positions +delta and at 0. Re-rotating the
    cached keys of the shifted run by -delta must reproduce the keys of the unshifted run, at
    *every* layer. Deeper layers see different hidden states only if attention is not purely
    relative, so this checks the whole network, not just layer 0. fp32 to isolate the math."""
    ids = text_ids[:, :128]
    n, delta = ids.shape[1], 3000
    caches = {}
    for off in (0, delta):
        cache = DynamicCache()
        pos = torch.arange(off, off + n, device=DEVICE)[None]
        model_fp32(ids, position_ids=pos, past_key_values=cache, use_cache=True)
        caches[off] = cache
    inv = model_fp32.model.rotary_emb.inv_freq
    for layer in range(len(caches[0].layers)):
        k0 = caches[0].layers[layer].keys
        k_shift = rerotate(caches[delta].layers[layer].keys, delta, inv)
        err = (k_shift - k0).abs().max().item()
        scale = k0.abs().max().item()
        assert err < 2e-3 * scale, (layer, err, scale)
        # values carry no position, so they must match too
        torch.testing.assert_close(caches[delta].layers[layer].values, caches[0].layers[layer].values,
                                   rtol=1e-3, atol=1e-3 * scale)
