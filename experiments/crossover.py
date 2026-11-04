"""Experiment 7: where does a sink cache start beating the full cache, at voice-call lengths?

Two questions, at session lengths of 2K, 4K, 8K and 16K tokens:

1. One session: per-token decode cost and KV memory, full cache vs sinks (v2).
   Per-token GPU time comes from latency.py (same execution mode for both:
   eager, summed kernel time); this script adds the memory side.
2. Many sessions on one GPU, which is how a voice service actually runs: the
   largest batch of concurrent sessions that fits in VRAM, and the aggregate
   decode throughput (tokens/s summed over sessions) at that batch. Every
   session in a batch is at the same length (lockstep), which is the best case
   for the full cache (no padding).

Batches are built by prefilling one session and replicating its cache along
the batch dimension: timing and memory depend on shapes, not on the text.

Realistic call lengths (for the README table), assuming ~150 spoken words per
minute across both speakers and ~1.3 tokens per word, i.e. ~200 tokens/min of
transcript, on top of a system prompt of ~1K tokens (instructions, tools,
caller profile):

    5-minute call  ~2K tokens     15-minute call ~4K
    30-minute call ~7K            60-minute call ~13K

    python experiments/crossover.py
"""

import argparse
import json
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from common import RESULTS, cache_len, kv_bytes_per_token, load_model, make_cache, pg19_tokens
from sinkcache import SinkCacheV2, StreamRunner

LENGTHS = [2048, 4096, 8192, 16384]
CONFIGS = ["full", "v2sink4_w1020"]
CHUNK = 256
WARMUP, TIMED, PROFILED = 5, 20, 3


def replicate(cache, batch: int):
    """Turn a batch-1 cache into `batch` identical sessions (new tensors; the old ones are freed)."""
    if isinstance(cache, SinkCacheV2):
        for l in cache.layers:
            for name in ("_kbuf", "_vbuf", "_reg_k"):
                setattr(l, name, getattr(l, name).expand(batch, -1, -1, -1).contiguous())
    else:
        for l in cache.layers:
            l.keys = l.keys.expand(batch, -1, -1, -1).contiguous()
            l.values = l.values.expand(batch, -1, -1, -1).contiguous()
    return cache


@torch.inference_mode()
def build(model, spec, ids, length, batch):
    cache = make_cache(spec, model)
    for s in range(0, length, CHUNK):
        model(ids[:, s:min(s + CHUNK, length)], past_key_values=cache, use_cache=True)
    return replicate(cache, batch)


@torch.inference_mode()
def time_steps(model, cache, batch, graph: bool):
    """Median wall-clock per decode step, and (eager only) summed GPU kernel time per step."""
    runner = StreamRunner(model, cache, use_graph=graph)
    tok = torch.full((batch, 1), 785, device="cuda")
    for _ in range(WARMUP):
        runner.step(tok)
    times = []
    for _ in range(TIMED):
        torch.cuda.synchronize()
        t = time.perf_counter()
        runner.step(tok)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    out = {"wall_ms": statistics.median(times)}
    if not graph:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILED):
                runner.step(tok)
            torch.cuda.synchronize()
        out["gpu_ms"] = sum(e.self_device_time_total for e in prof.key_averages()
                            if e.device_type.name == "CUDA") / PROFILED / 1e3
    runner.graph = None
    return out


def fits(model, spec, ids, length, batch) -> bool:
    try:
        cache = build(model, spec, ids, length, batch)
        with torch.inference_mode():
            for _ in range(3):  # a few steps: the full cache's torch.cat needs 2x its size transiently
                model(torch.full((batch, 1), 785, device="cuda"), past_key_values=cache, use_cache=True)
        ok = True
    except torch.OutOfMemoryError:
        ok = False
    cache = None
    torch.cuda.empty_cache()
    return ok


def max_batch(model, spec, ids, length, hi=256) -> int:
    lo = 0  # largest known to fit
    b = 1
    while b <= hi and fits(model, spec, ids, length, b):  # exponential search, then bisect
        lo, b = b, b * 2
    hi = min(b, hi + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(model, spec, ids, length, mid) else (lo, mid)
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--lengths", type=int, nargs="*", default=LENGTHS)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 4, 16])
    args = ap.parse_args()

    model, tok = load_model(args.model)
    ids = pg19_tokens(tok, args.model, 65536).cuda()
    kv = kv_bytes_per_token(model.config)
    rows = []
    for length in args.lengths:
        for spec in CONFIGS:
            base = torch.cuda.memory_allocated()
            cache = build(model, spec, ids, length, 1)
            kv_mib = (torch.cuda.memory_allocated() - base) / 2**20
            held = cache_len(cache)
            cache = None
            torch.cuda.empty_cache()
            bmax = max_batch(model, spec, ids, length)
            row = {"length": length, "config": spec, "kv_mib_per_session": round(kv_mib, 1),
                   "tokens_held": held, "max_sessions": bmax, "batches": {}}
            for b in sorted(set(args.batches + [bmax])):
                if b < 1 or b > bmax:
                    continue
                cache = build(model, spec, ids, length, b)
                r = time_steps(model, cache, b, graph=False)
                if spec != "full":
                    r["graph_wall_ms"] = time_steps(model, cache, b, graph=True)["wall_ms"]
                r["tok_per_s_gpu"] = b / r["gpu_ms"] * 1e3
                if "graph_wall_ms" in r:
                    r["tok_per_s_graph"] = b / r["graph_wall_ms"] * 1e3
                row["batches"][b] = r
                cache = None
                torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), flush=True)

    out = {"model": args.model, "gpu": torch.cuda.get_device_name(0),
           "total_vram_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
           "kv_bytes_per_token": kv, "rows": rows}
    path = RESULTS / f"crossover_{args.model}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
