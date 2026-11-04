"""SinkCacheV2 (sinks moved forward, window untouched) against DynamicCache, against an independent
reference built from pre-RoPE keys, against v1, and graph replay against eager."""

import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from sinkcache import SinkCache, SinkCacheV2, StreamRunner

from .conftest import DEVICE
from .test_cache import PreRopeRecorder, stream


@torch.no_grad()
def reference_from_kept(model, rec, cache):
    """DynamicCache holding the tokens v2 says it keeps, each pre-RoPE key rotated *once* by HF's
    rotary code at the position v2 says it sits at (region: before the window; window: absolute)."""
    layer0 = cache.layers[0]
    t = layer0.t
    ws = layer0._window_start(t)
    kept_pos = list(layer0.region_pos) + list(range(max(ws, layer0.n_sink), t))
    at = torch.tensor(layer0.kept()[2], device=DEVICE)[None]
    idx = torch.tensor(kept_pos, device=DEVICE)
    ref = DynamicCache()
    for i in range(model.config.num_hidden_layers):
        k, v = rec.get(i)
        k, v = k[:, :, idx], v[:, :, idx]
        cos, sin = model.model.rotary_emb(k, at)
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        ref.update(k, v, i)
    return ref, kept_pos


def check(model, ids, n_sink, window, prefill=16, pins=(), max_pinned=0, plan=None):
    rec = PreRopeRecorder(model)
    cache = SinkCacheV2.from_model(model, n_sink=n_sink, window=window, max_pinned=max_pinned)
    for a, b in pins:
        cache.pin(a, b)
    if plan is None:
        stream(model, ids[:, :-1], cache, prefill=prefill)
    else:
        t = 0
        with torch.no_grad():
            for n in plan:
                model(ids[:, t:t + n], past_key_values=cache, use_cache=True)
                t += n
        assert t == ids.shape[1] - 1
    rec.remove()
    ref, kept_pos = reference_from_kept(model, rec, cache)
    snap = [(l.kept()[0].clone(), l.kept()[1].clone(), r.keys.clone(), r.values.clone())
            for l, r in zip(cache.layers, ref.layers)]
    t = cache.tokens_seen
    pos = torch.tensor([[t]], device=DEVICE)
    with torch.no_grad():
        got = model(ids[:, -1:], past_key_values=cache, use_cache=True).logits.float()
        want = model(ids[:, -1:], past_key_values=ref, use_cache=True, position_ids=pos).logits.float()
    return got, want, cache, snap, kept_pos


def test_no_eviction_matches_dynamic_cache_bitwise(model_bf16, text_ids):
    ids = text_ids[:, :200]
    ref = stream(model_bf16, ids, DynamicCache())
    got = stream(model_bf16, ids, SinkCacheV2.from_model(model_bf16, n_sink=4, window=1020))
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("n_sink", [0, 1, 4])
def test_matches_reference_fp32(model_fp32, text_ids, n_sink):
    got, want, cache, snap, kept = check(model_fp32, text_ids[:, :301], n_sink, 60)
    assert len(kept) == n_sink + 60 and cache.tokens_evicted == 301 - n_sink - 60  # after the final step
    assert cache.get_seq_length() == 301  # absolute: the last step's query sat at position 300
    for ka, va, kb, vb in snap:
        torch.testing.assert_close(va, vb, rtol=0, atol=0)
        # sink keys: v2 computes the rotation angle in fp64, HF in fp32 (position * theta rounded)
        torch.testing.assert_close(ka, kb, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got, want, rtol=0, atol=2e-3)


def test_bf16_window_keys_are_exact(model_bf16, text_ids):
    """No drift: window keys are bit-for-bit what the model wrote, after 236 evictions; sink keys
    are within one bf16 rounding of being rotated directly at their new position."""
    got, want, cache, snap, _ = check(model_bf16, text_ids[:, :301], 4, 60)
    for ka, va, kb, vb in snap:
        torch.testing.assert_close(ka[:, :, 4:], kb[:, :, 4:], rtol=0, atol=0)
        torch.testing.assert_close(va, vb, rtol=0, atol=0)
        torch.testing.assert_close(ka[:, :, :4].float(), kb[:, :, :4].float(), rtol=2 ** -7,
                                   atol=2 ** -7 * kb.float().abs().max().item())
    assert (got.argmax(-1) == want.argmax(-1)).all()


def test_matches_v1(model_fp32, text_ids):
    """Same relative positions as v1 (re-rotated window, fp32 keys): same logits up to rounding."""
    ids = text_ids[:, :400]
    v1 = stream(model_fp32, ids, SinkCache.from_model(model_fp32, 4, 60))
    v2 = stream(model_fp32, ids, SinkCacheV2.from_model(model_fp32, 4, 60))
    torch.testing.assert_close(v2, v1, rtol=0, atol=1e-3)


def test_chunks_and_long_prefill(model_fp32, text_ids):
    """Chunks larger than the window, and a prefill longer than the whole cache."""
    ids = text_ids[:, :401]
    got, want, *_ = check(model_fp32, ids, 4, 60, plan=[200, 1, 1, 37, 90, 1, 1, 69])
    torch.testing.assert_close(got, want, rtol=0, atol=2e-3)
    with torch.no_grad():
        dyn = model_fp32(ids[:, :200], past_key_values=DynamicCache(), use_cache=True).logits
        chunk = model_fp32(ids[:, :200], past_key_values=SinkCacheV2.from_model(model_fp32, 4, 60),
                           use_cache=True).logits
    torch.testing.assert_close(chunk, dyn, rtol=0, atol=0)


def test_pinned_tokens_move_to_region(model_fp32, text_ids):
    """Pinned spans survive eviction, sit after the sinks in order, and the cache still equals
    attention over exactly the kept tokens at the claimed positions."""
    pins = [(20, 30), (100, 105)]
    got, want, cache, _, kept = check(model_fp32, text_ids[:, :301], 4, 60, pins=pins, max_pinned=32,
                                      plan=[16] + [1] * 84 + [40] + [1] * 160)
    assert kept[:19] == [0, 1, 2, 3] + list(range(20, 30)) + list(range(100, 105))
    assert cache.region_size == 19
    torch.testing.assert_close(got, want, rtol=0, atol=2e-3)


def test_pin_budget_drops_oldest(model_bf16, text_ids):
    cache = SinkCacheV2.from_model(model_bf16, n_sink=4, window=60, max_pinned=8)
    cache.pin(10, 16)
    cache.pin(30, 36)
    stream(model_bf16, text_ids[:, :200], cache)
    layer = cache.layers[0]
    assert layer.region_pos == [0, 1, 2, 3, 14, 15, 30, 31, 32, 33, 34, 35] and layer.pins_dropped == 4


@pytest.mark.parametrize("pinned", [False, True])
def test_cuda_graph_replay_matches_eager(model_bf16, text_ids, pinned):
    """Graphed steady steps (growing positions from a device counter), chunks in between, and
    pin moves forcing eager steps + recapture: bitwise equal to all-eager."""
    ids = text_ids[:, :400]
    plan = [16] + [1] * 120 + [30] + [1] * 80 + [45] + [1] * 100
    outs, replays = {}, {}
    for graphed in (False, True):
        cache = SinkCacheV2.from_model(model_bf16, n_sink=4, window=60, max_pinned=16 if pinned else 0)
        if pinned:
            cache.pin(140, 146)
            cache.pin(200, 203)
        runner = StreamRunner(model_bf16, cache, use_graph=graphed)
        out, t = [], 0
        with torch.no_grad():
            for n in plan:
                out.append(runner.step(ids[:, t:t + n]).float().clone())
                t += n
        outs[graphed], replays[graphed] = torch.cat(out, 1), runner.replays
        assert cache.tokens_seen == sum(plan)
        if pinned:
            assert cache.region_size == 4 + 9
    assert replays[True] > 200
    torch.testing.assert_close(outs[True], outs[False], rtol=0, atol=0)


def test_snapshot_restore(model_bf16, text_ids):
    ids = text_ids[:, :300]
    cache = SinkCacheV2.from_model(model_bf16, 4, 60)
    stream(model_bf16, ids[:, :200], cache)
    snap = cache.snapshot()
    with torch.no_grad():
        a = model_bf16(ids[:, 200:260], past_key_values=cache, use_cache=True).logits
        cache.restore(snap)
        b = model_bf16(ids[:, 200:260], past_key_values=cache, use_cache=True).logits
    torch.testing.assert_close(a, b, rtol=0, atol=0)
