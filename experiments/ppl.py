"""Experiments 1 and 2: streaming perplexity and memory against position.

PG19 test text is fed one token at a time through the cache. The NLL of every
next token, plus allocated / peak GPU memory, is written to CSV every 1,024
tokens, so a crash or OOM keeps what was measured. One config per process
(clean allocator state, clean OOM).

Sink configs switch to CUDA-graph replay once the cache is full (bitwise
identical to eager, see tests); the full cache runs eagerly.

    python experiments/ppl.py --config sink4_w1020 --tokens 65536
    python experiments/ppl.py --config full --tokens 65536     # runs until OOM
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F

from common import RESULTS, kv_bytes_per_token, load_model, make_cache, pg19_tokens
from sinkcache import StreamRunner

FLUSH = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    ids = pg19_tokens(tok, args.model, 65536)[:, :args.tokens].cuda()
    out_dir = RESULTS / "ppl" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{args.config}.csv"
    mem_csv = out_dir / f"{args.config}_mem.csv"
    csv.write_text("pos,nll\n")
    mem_csv.write_text("tokens,allocated_mib,peak_mib,reserved_mib\n")

    torch.cuda.reset_peak_memory_stats()
    base_mib = torch.cuda.memory_allocated() / 2**20
    cache = make_cache(args.config, model)
    runner = StreamRunner(model, cache, use_graph=not args.no_graph)
    n = ids.shape[1]
    buf = torch.empty(FLUSH, device="cuda")
    written, oom_at, t0 = 0, None, time.time()

    def flush(upto):
        nonlocal written
        vals = buf[: upto - written].tolist()
        with csv.open("a") as f:
            f.writelines(f"{written + i + 1},{v:.5f}\n" for i, v in enumerate(vals))
        with mem_csv.open("a") as f:
            f.write(f"{upto},{torch.cuda.memory_allocated() / 2**20:.1f},"
                    f"{torch.cuda.max_memory_allocated() / 2**20:.1f},"
                    f"{torch.cuda.memory_reserved() / 2**20:.1f}\n")
        written = upto
        # A growing DynamicCache frees and re-requests slightly larger blocks every step, which
        # the caching allocator cannot reuse; hand them back so reserved tracks allocated.
        if not runner.use_graph:
            torch.cuda.empty_cache()

    with torch.inference_mode():
        for t in range(n - 1):
            try:
                logits = runner.step(ids[:, t:t + 1])[0, -1]
                # NLL of token t+1 given tokens 0..t; "pos" in the CSV is t+1 (context length)
                buf[t - written] = F.cross_entropy(logits.float()[None], ids[0, t + 1:t + 2])
            except torch.OutOfMemoryError:
                oom_at = t
                break
            if t + 1 - written == FLUSH:
                flush(t + 1)
                rate = (t + 1) / (time.time() - t0)
                print(f"{args.config}: {t + 1}/{n - 1}  {rate:.1f} tok/s  "
                      f"alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB  "
                      f"graph={runner.graph is not None}", flush=True)
    if oom_at is not None:
        print(f"{args.config}: OOM at {oom_at} tokens of context", flush=True)
        flush(oom_at)
        del cache, runner
        torch.cuda.empty_cache()
    elif written < n - 1:
        flush(n - 1)

    meta = {
        "config": args.config, "model": args.model, "tokens": n, "oom_at": oom_at,
        "seconds": round(time.time() - t0, 1), "graph": not args.no_graph,
        "weights_mib": round(base_mib, 1),
        "kv_bytes_per_token": kv_bytes_per_token(model.config),
        "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "total_vram_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
    }
    (out_dir / f"{args.config}.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
