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
    assert cache.get_seq_length() == 64          # logical: the next token's position
    assert cache.is_steady
    # 300 seen; 65 physically held (64 kept + 1 pending eviction, applied at the next update)
    assert cache.tokens_seen == 300 and cache.tokens_evicted == 235
    for layer in cache.layers:
        k, v = layer.kept()
        assert k.shape[-2] == v.shape[-2] == 64
        assert layer._kbuf.shape[-2] == 65       # buffer never grows past n_sink + window + 1


@pytest.mark.skipif(DEVICE != "cuda", reason="triton kernel")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_sink,window,drop", [(4, 60, 1), (0, 60, 1), (4, 1020, 1), (4, 100, 37), (2, 7, 200)])
def test_triton_eviction_matches_torch(dtype, n_sink, window, drop):
    """Fused in-place shift + re-rotation == the PyTorch path (K to within one rounding, V bit-exact),
    including drops larger than the chunk size and windows smaller than it."""
    from sinkcache.cache import SinkLayer
    torch.manual_seed(drop)
    inv = 1.0 / (1e6 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    k = torch.randn(1, 8, n_sink + window + drop, 128, device=DEVICE, dtype=dtype) * 30
    v = torch.randn_like(k)
    out = {}
    for fused in (False, True):
        layer = SinkLayer(n_sink, window, inv, use_triton=fused)
        layer.update(k, v)            # one chunk: n_sink + window + drop tokens, `drop` pending
        layer._evict_pending()
        out[fused] = (layer._kbuf[:, :, :n_sink + window].float(), layer._vbuf[:, :, :n_sink + window])
    (kt, vt), (kf, vf) = out[False], out[True]
    torch.testing.assert_close(vf, vt, rtol=0, atol=0)
    # one output rounding apart (FMA vs mul+add); relative to the inputs' scale, since an
    # output near zero by cancellation has no meaningful elementwise relative error
    ulp = 2 ** -7 if dtype == torch.bfloat16 else 2 ** -22
    torch.testing.assert_close(kf, kt, rtol=ulp, atol=ulp * kt.abs().max().item())


def test_fast_rerotate_matches_reference():
    from sinkcache.rope import rerotate
    from sinkcache.cache import SinkLayer
    torch.manual_seed(0)
    inv = 1.0 / (1e6 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    layer = SinkLayer(4, 60, inv.to(DEVICE))
    layer.device = DEVICE
    layer.inv_freq = inv.to(DEVICE)
    k = torch.randn(1, 8, 60, 128, device=DEVICE, dtype=torch.bfloat16)
    for d in (1, 5, 300):
        torch.testing.assert_close(layer._rerotate(k, d).to(k.dtype), rerotate(k, d, inv), rtol=0, atol=0)


# ------------------------------------------------------------------ after eviction

def _eviction_check(model, ids, n_sink, window, rerotate_keys=True, prefill=16, use_triton=None):
    """Next-token logits from the sink cache vs. the kept-tokens reference."""
    rec = PreRopeRecorder(model)
    cache = SinkCache.from_model(model, n_sink=n_sink, window=window, rerotate_keys=rerotate_keys,
                                 use_triton=use_triton)
    stream(model, ids[:, :-1], cache, prefill=prefill)
    rec.remove()
    ref = kept_tokens_reference(model, rec, ids.shape[1] - 1, n_sink, window)
    # snapshot both caches' contents before the next step mutates them
    snap = [(*[t.clone() for t in a.kept()], b.keys.clone(), b.values.clone())
            for a, b in zip(cache.layers, ref.layers)]
    nxt = ids[:, -1:]
    with torch.no_grad():
        got = model(nxt, past_key_values=cache, use_cache=True).logits.float()
        want = model(nxt, past_key_values=ref, use_cache=True).logits.float()
    return got, want, cache, snap


@pytest.mark.parametrize("use_triton", [False, True], ids=["torch", "triton"])
@pytest.mark.parametrize("n_sink", [0, 4])
def test_eviction_equals_attention_over_kept_tokens_fp32(model_fp32, text_ids, n_sink, use_triton):
    """fp32 isolates the math: after 230+ evictions (each window key re-rotated up to 60 times)
    the cache is the kept tokens at contiguous positions."""
    got, want, cache, snap = _eviction_check(model_fp32, text_ids[:, :301], n_sink, 60, use_triton=use_triton)
    assert cache.tokens_evicted > 200 and cache.layers[0].use_triton == use_triton
    err = (got - want).abs().max().item()
    kerr = max(((ka - kb).norm() / kb.norm()).item() for ka, _, kb, _ in snap)
    print(f"fp32 n_sink={n_sink} triton={use_triton}: max|dlogit|={err:.2e} key rel err={kerr:.2e}")
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


def test_cuda_graph_replay_matches_eager(model_bf16, text_ids):
    """The graphed steady-state step runs the same kernels on the same buffers: bitwise equal."""
    from sinkcache import StreamRunner
    ids = text_ids[:, :260]
    eager = stream(model_bf16, ids, SinkCache.from_model(model_bf16, n_sink=4, window=60))
    cache = SinkCache.from_model(model_bf16, n_sink=4, window=60)
    runner = StreamRunner(model_bf16, cache, use_graph=True)
    with torch.no_grad():
        out = [model_bf16(ids[:, :16], past_key_values=cache, use_cache=True).logits.float()]
        for t in range(16, ids.shape[1]):
            out.append(runner.step(ids[:, t:t + 1]).float().clone())
    assert runner.graph is not None
    assert cache.tokens_seen == 260 and cache.tokens_evicted == 260 - 65
    torch.testing.assert_close(torch.cat(out, 1), eager, rtol=0, atol=0)
