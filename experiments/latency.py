"""Experiment 3: per-token decode latency against context length.

For each round (config order rotated every round, as laptop clocks drift) and
each config, the cache is built by chunked prefill of PG19 text, and at each
checkpoint length we measure one-token decode steps:

  eager_ms   wall-clock per token, eager PyTorch (what plain HF gives on this machine)
  gpu_ms     sum of CUDA kernel durations per token (CUPTI via torch.profiler): the
             hardware cost, comparable to the bytes-moved model below
  graph_ms   wall-clock per token with the step replayed as a CUDA graph (sink cache only:
             a growing DynamicCache cannot be captured)

plus a per-category kernel-time breakdown (linear layers / attention / cache copies / rest).
Reported: median over rounds of the per-round medians, and the range over rounds.

    python experiments/latency.py --rounds 3
"""

import argparse
import json
import statistics
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from common import RESULTS, kv_bytes_per_token, load_model, make_cache, pg19_tokens
from sinkcache import StreamRunner

POSITIONS = [1024, 2048, 4096, 8192, 16384, 32768]
CONFIGS = ["full", "sink4_w1020"]
CHUNK = 256
WARMUP, TIMED, PROFILED = 10, 30, 5


def categorize(name: str) -> str:
    n = name.lower()
    if "gemv" in n or "gemm" in n or "cutlass" in n or "sm80" in n or "sm90" in n or "ampere" in n:
        return "linear"
    if "decode_partial" in n or "decode_reduce" in n or "softmax" in n or "fmha" in n or "attention" in n:
        return "attention"
    if "cat" in n or "copy" in n or "shift_kernel" in n:
        return "cache_copy"
    return "other"


@torch.inference_mode()
def measure(model, cache, tok_ids, graph: bool):
    """Times one-token steps feeding tok_ids[i] (a fixed real token per step is fine for timing)."""
    runner = StreamRunner(model, cache, use_graph=graph)
    step = lambda: runner.step(tok_ids)
    for _ in range(WARMUP):
        step()
    times = []
    for _ in range(TIMED):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    out = {"ms": statistics.median(times)}
    if not graph:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILED):
                step()
            torch.cuda.synchronize()
        cats = defaultdict(float)
        for e in prof.key_averages():
            if e.device_type.name == "CUDA" and e.self_device_time_total > 0:
                cats[categorize(e.key)] += e.self_device_time_total / PROFILED / 1e3
        out["gpu_ms"] = sum(cats.values())
        out["breakdown"] = dict(cats)
    runner.graph = None
    return out


@torch.inference_mode()
def run_config(model, spec, ids, positions):
    cache = make_cache(spec, model)
    rows, done = [], 0
    tok = ids[:, :1]
    for p in positions:
        while done < p:
            n = min(CHUNK, p - done)
            model(ids[:, done:done + n], past_key_values=cache, use_cache=True)
            done += n
        eager = measure(model, cache, tok, graph=False)
        row = {"config": spec, "position": p, "cache_len": cache.get_seq_length(),
               "eager_ms": eager["ms"], "gpu_ms": eager["gpu_ms"], "breakdown": eager["breakdown"]}
        if spec != "full":
            row["graph_ms"] = measure(model, cache, tok, graph=True)["ms"]
        done = cache.get_seq_length() if spec == "full" else done  # full cache grew by the timed steps
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k != "breakdown"},
              flush=True)
    del cache
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--positions", type=int, nargs="*", default=POSITIONS)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    ids = pg19_tokens(tok, args.model, 65536).cuda()
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    rows = []
    for r in range(args.rounds):
        order = CONFIGS[r % len(CONFIGS):] + CONFIGS[:r % len(CONFIGS)]
        for spec in order:
            for row in run_config(model, spec, ids, args.positions):
                rows.append({"round": r, **row})

    summary = []
    for spec in CONFIGS:
        for p in args.positions:
            rs = [x for x in rows if x["config"] == spec and x["position"] == p]
            entry = {"config": spec, "position": p, "cache_len": rs[0]["cache_len"]}
            for key in ("eager_ms", "gpu_ms", "graph_ms"):
                vals = [x[key] for x in rs if key in x]
                if vals:
                    entry[key] = statistics.median(vals)
                    entry[key + "_range"] = [min(vals), max(vals)]
            cats = defaultdict(list)
            for x in rs:
                for c, v in x["breakdown"].items():
                    cats[c].append(v)
            entry["breakdown_ms"] = {c: statistics.median(v) for c, v in cats.items()}
            summary.append(entry)

    out = {
        "model": args.model, "rounds": args.rounds, "weight_bytes": weight_bytes,
        "kv_bytes_per_token": kv_bytes_per_token(model.config),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "summary": summary, "rows": rows,
    }
    path = RESULTS / f"latency_{args.model}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
