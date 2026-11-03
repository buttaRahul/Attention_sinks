"""Triton flash-decode vs an fp32 reference, and as a Transformers attention backend."""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache

from sinkcache import SinkCache
from sinkcache.decode_attention import calls, flash_decode, register

from .conftest import DEVICE, MODEL

pytestmark = pytest.mark.skipif(DEVICE != "cuda", reason="triton kernel")


def ref_attention(q, k, v, scale):
    g = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(g, 1)
    v = v.float().repeat_interleave(g, 1)
    return F.scaled_dot_product_attention(q.float(), k, v, scale=scale).transpose(1, 2)


@pytest.mark.parametrize("hq,hkv,d", [(16, 8, 128), (32, 8, 64), (8, 8, 128), (14, 2, 64)])
@pytest.mark.parametrize("seq_len", [1, 63, 64, 1025, 4097, 32768])
def test_flash_decode_matches_reference(hq, hkv, d, seq_len):
    torch.manual_seed(seq_len)
    q = torch.randn(1, hq, 1, d, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, hkv, seq_len, d, device=DEVICE, dtype=torch.bfloat16) * 3
    v = torch.randn_like(k)
    out = flash_decode(q, k, v, d ** -0.5).float()
    torch.testing.assert_close(out, ref_attention(q, k, v, d ** -0.5), rtol=2e-2, atol=1e-2)


def test_flash_decode_strided_view():
    """SinkLayer hands attention a view into a larger buffer (non-contiguous over heads)."""
    buf_k = torch.randn(2, 8, 1100, 128, device=DEVICE, dtype=torch.bfloat16)
    buf_v = torch.randn_like(buf_k)
    k, v = buf_k[:, :, :1025], buf_v[:, :, :1025]
    q = torch.randn(2, 16, 1, 128, device=DEVICE, dtype=torch.bfloat16)
    torch.testing.assert_close(flash_decode(q, k, v, 0.088).float(), ref_attention(q, k, v, 0.088),
                               rtol=2e-2, atol=1e-2)


@torch.no_grad()
def test_backend_in_model(model_bf16, text_ids):
    """Qwen3 with the registered backend: decode goes through the kernel, prefill through SDPA, and
    greedy next tokens agree with plain SDPA over 64 steps of a sink cache."""
    name = register()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=name)
    model = model.to(DEVICE).eval()
    ids = text_ids[:, :400]
    for make in (DynamicCache, lambda: SinkCache.from_model(model, 4, 124)):
        logits = {}
        for m in (model_bf16, model):
            cache = make() if make is DynamicCache else SinkCache.from_model(m, 4, 124)
            out = [m(ids[:, :200], past_key_values=cache, use_cache=True).logits[:, -1:]]
            for t in range(200, 264):
                out.append(m(ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits)
            logits[m is model] = torch.cat(out, 1).float()
        a, b = logits[False], logits[True]
        assert (a.argmax(-1) == b.argmax(-1)).float().mean() > 0.97
        assert (a - b).abs().max() < 1.0  # bf16 noise between two attention kernels
    assert calls["decode_kernel"] > 0 and calls["sdpa_fallback"] > 0
