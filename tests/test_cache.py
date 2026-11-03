"""Correctness of SinkCache against DynamicCache and against an independently built reference."""

import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from sinkcache import SinkCache

from .conftest import DEVICE


class PreRopeRecorder:
    """Records each layer's keys *before* RoPE (after k_norm for Qwen3) and values, for every token fed."""

    def __init__(self, model):
        self.keys = {i: [] for i in range(model.config.num_hidden_layers)}
        self.values = {i: [] for i in range(model.config.num_hidden_layers)}
        self.handles = []
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            hd = attn.head_dim
            k_src = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
            self.handles.append(k_src.register_forward_hook(self._hook(self.keys[i], hd)))
            self.handles.append(attn.v_proj.register_forward_hook(self._hook(self.values[i], hd)))

    @staticmethod
    def _hook(store, hd):
        def fn(_mod, _inp, out):
            b, q = out.shape[:2]
            store.append(out.reshape(b, q, -1, hd).transpose(1, 2).detach())
        return fn

    def get(self, layer):
        return torch.cat(self.keys[layer], dim=-2), torch.cat(self.values[layer], dim=-2)

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def stream(model, ids, cache, prefill=16):
    """Prefill `prefill` tokens in one chunk, then feed the rest one at a time. Returns per-token logits."""
    out = [model(ids[:, :prefill], past_key_values=cache, use_cache=True).logits]
    for t in range(prefill, ids.shape[1]):
        out.append(model(ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits)
    return torch.cat(out, dim=1).float()


@torch.no_grad()
def kept_tokens_reference(model, recorder, n_tokens, n_sink, window):
    """A DynamicCache holding exactly the kept tokens' K/V, rotated *directly* (once, by HF's own
    rotary code) at their new contiguous positions 0..cap-1. No re-rotation involved."""
    kept = list(range(n_sink)) + list(range(n_tokens - window, n_tokens))
    idx = torch.tensor(kept, device=DEVICE)
    pos = torch.arange(len(kept), device=DEVICE)[None]
    ref = DynamicCache()
    for layer in range(model.config.num_hidden_layers):
        k, v = recorder.get(layer)
        k, v = k[:, :, idx], v[:, :, idx]
        cos, sin = model.model.rotary_emb(k, pos)
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        ref.update(k, v, layer)
    return ref


# ------------------------------------------------------------------ before any eviction

def test_no_eviction_matches_dynamic_cache_bitwise(model_bf16, text_ids):
    ids = text_ids[:, :200]
    ref = stream(model_bf16, ids, DynamicCache())
    sink = stream(model_bf16, ids, SinkCache.from_model(model_bf16, n_sink=4, window=1020))
    torch.testing.assert_close(sink, ref, rtol=0, atol=0)


def test_cache_size_is_bounded(model_bf16, text_ids):
    cache = SinkCache.from_model(model_bf16, n_sink=4, window=60)
    stream(model_bf16, text_ids[:, :300], cache)
    assert cache.get_seq_length() == 64
    assert cache.tokens_seen == 300 and cache.tokens_evicted == 236
    for layer in cache.layers:
        assert layer.keys.shape[-2] == layer.values.shape[-2] == 64


# ------------------------------------------------------------------ after eviction

def _eviction_check(model, ids, n_sink, window, rerotate_keys=True, prefill=16):
    """Next-token logits from the sink cache vs. the kept-tokens reference."""
    rec = PreRopeRecorder(model)
    cache = SinkCache.from_model(model, n_sink=n_sink, window=window, rerotate_keys=rerotate_keys)
    stream(model, ids[:, :-1], cache, prefill=prefill)
    rec.remove()
    ref = kept_tokens_reference(model, rec, ids.shape[1] - 1, n_sink, window)
    # snapshot both caches' contents before the next step mutates them
    snap = [(a.keys.clone(), a.values.clone(), b.keys.clone(), b.values.clone())
            for a, b in zip(cache.layers, ref.layers)]
    nxt = ids[:, -1:]
    with torch.no_grad():
        got = model(nxt, past_key_values=cache, use_cache=True).logits.float()
        want = model(nxt, past_key_values=ref, use_cache=True).logits.float()
    return got, want, cache, snap


@pytest.mark.parametrize("n_sink", [0, 4])
def test_eviction_equals_attention_over_kept_tokens_fp32(model_fp32, text_ids, n_sink):
    """fp32 isolates the math: after 230+ evictions (each window key re-rotated up to 60 times)
    the cache is the kept tokens at contiguous positions."""
    got, want, cache, snap = _eviction_check(model_fp32, text_ids[:, :301], n_sink, 60)
    assert cache.tokens_evicted > 200
    err = (got - want).abs().max().item()
    kerr = max(((ka - kb).norm() / kb.norm()).item() for ka, _, kb, _ in snap)
    print(f"fp32 n_sink={n_sink}: max|dlogit|={err:.2e} key rel err={kerr:.2e}")
    for ka, va, kb, vb in snap:
        torch.testing.assert_close(va, vb, rtol=0, atol=0)  # values are only sliced, never touched
    assert kerr < 1e-5
    torch.testing.assert_close(got, want, rtol=0, atol=2e-3)


@pytest.mark.parametrize("n_sink", [0, 4])
def test_eviction_equals_attention_over_kept_tokens_bf16(model_bf16, text_ids, n_sink):
    """bf16: same check; the difference is re-rotation rounding accumulated in bf16 storage."""
    got, want, cache, snap = _eviction_check(model_bf16, text_ids[:, :301], n_sink, 60)
    err = (got - want).abs().max().item()
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    kerr = max(((ka.float() - kb.float()).norm() / kb.float().norm()).item() for ka, _, kb, _ in snap)
    print(f"bf16 n_sink={n_sink}: max|dlogit|={err:.3f} key rel err={kerr:.2e} top1={agree}")
    assert agree == 1.0
    assert kerr < 2e-2
    assert err < 0.5


def test_stale_rotation_is_caught(model_fp32, text_ids):
    """Teeth: without re-rotation the kept keys sit at stale positions and the check fails badly."""
    got, want, _, _ = _eviction_check(model_fp32, text_ids[:, :301], 4, 60, rerotate_keys=False)
    assert (got - want).abs().max().item() > 1.0


def test_prefill_longer_than_cache(model_fp32, text_ids):
    """A single chunk longer than the cache: the chunk itself gets full causal attention (identical
    to DynamicCache), then the cache is trimmed and the next token sees only the kept tokens."""
    ids = text_ids[:, :201]
    with torch.no_grad():
        dyn = model_fp32(ids[:, :200], past_key_values=DynamicCache(), use_cache=True).logits
    got, want, cache, _ = _eviction_check(model_fp32, ids, 4, 60, prefill=200)
    assert cache.get_seq_length() == 64
    torch.testing.assert_close(got, want, rtol=0, atol=2e-3)
    with torch.no_grad():
        chunk = model_fp32(ids[:, :200], past_key_values=SinkCache.from_model(model_fp32, 4, 60),
                           use_cache=True).logits
    torch.testing.assert_close(chunk, dyn, rtol=0, atol=0)
