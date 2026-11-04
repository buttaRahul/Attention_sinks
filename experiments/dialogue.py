"""Experiment 6: perplexity and fact recall on real multi-session dialogue (LoCoMo).

LoCoMo (Maharana et al., 2024; github.com/snap-research/locomo) has 10 long
conversations between two people, each spanning up to ~35 sessions over
months, with ~200 annotated questions per conversation. Each question lists
its evidence turns ("D3:5" = session 3, turn 5), so for every question we know
how far back its answer was said, and whether that is still inside the cache.

Per conversation and config:

  1. The transcript is streamed after a short system prompt, one line per
     turn, "[session date] Speaker: text". Sink caches go one token at a time
     (graph-replayed); the full cache is fed in 512-token chunks, which for a
     full cache gives exactly the same NLL as one token at a time. Per-token
     NLL is saved, so perplexity on dialogue can be binned by position.
  2. After the transcript, every question in categories 1-4 (single-hop,
     multi-hop, temporal, open-domain; category 5 is adversarial/unanswerable)
     is asked on a checkpoint of the cache, then the cache is restored, so
     questions don't see each other. Recorded per question:
       - teacher-forced mean log p of the gold answer;
       - the greedy answer, and `--seeds` sampled answers (T=0.7), scored with
         LoCoMo's token F1;
       - distance from the question to its evidence, in tokens, and whether
         the evidence is still in the window.

Uncertainty comes from the ~1,500 questions over 10 conversations (bootstrap
over conversations in plot.py), and from several sampled answers per question.

    python experiments/dialogue.py --config full
    python experiments/dialogue.py --config v2sink4_w1020
    python experiments/dialogue.py --config v2sink0_w1024     # window only
"""

import argparse
import json
import re
import sys
import time
import urllib.request

import torch
import torch.nn.functional as F

from common import (DATA, RESULTS, Checkpoint, answer_logprob, cache_len, chat_turn, generate, load_model,
                    make_cache, token_f1)
from sinkcache import SinkCacheV2, StreamRunner

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
SYSTEM = ("You are a memory assistant. Below is the full chat history between two friends, across "
          "many sessions. Afterwards you will answer questions about it in a few words.")
QUESTION = "Based on the conversation above, answer in a few words: {q}"
FULL_CHUNK = 512
MAX_ANSWER = 24


def load_locomo():
    path = DATA / "locomo10.json"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return json.loads(path.read_text(encoding="utf-8"))


def transcript(conv: dict):
    """Lines in session order, each with its dia_id."""
    c = conv["conversation"]
    sessions = sorted((int(m.group(1)), k) for k in c if (m := re.fullmatch(r"session_(\d+)", k)))
    for n, key in sessions:
        date = c.get(f"session_{n}_date_time", "")
        for i, turn in enumerate(c[key]):
            head = f"\n[{date}]\n" if i == 0 else ""
            yield turn["dia_id"], f"{head}{turn['speaker']}: {turn['text']}\n"


@torch.inference_mode()
def stream_ids(model, cache, runner, ids, nll, full, last=None):
    """Feed ids (1, n) after whatever is in the cache; append the NLL of every token to nll (given
    `last`, the logits that predict ids[0]). Returns the logits for the token after the last one."""
    n = ids.shape[1]
    if full:
        for s in range(0, n, FULL_CHUNK):
            chunk = ids[:, s:s + FULL_CHUNK]
            logits = model(chunk, past_key_values=cache, use_cache=True).logits[0].float()
            if last is not None:
                nll.append(F.cross_entropy(last[None], chunk[0, :1]).item())
            if chunk.shape[1] > 1:
                nll.extend(F.cross_entropy(logits[:-1], chunk[0, 1:], reduction="none").tolist())
            last = logits[-1]
        return last
    for i in range(n):
        logits = runner.step(ids[:, i:i + 1])[0, -1].float()
        if last is not None:
            nll.append(F.cross_entropy(last[None], ids[0, i:i + 1]).item())
        last = logits.clone()
    return last


@torch.inference_mode()
def run_conversation(model, tok, conv, spec, seeds, window):
    cache = make_cache(spec, model)
    full = spec == "full"
    runner = None if full else StreamRunner(model, cache, use_graph=True)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    # system + opening of the user message, then the transcript line by line
    head = tok(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n", return_tensors="pt",
               add_special_tokens=False).input_ids.cuda()
    pos_of, nll, line_start = {}, [], head.shape[1]
    last = model(head, past_key_values=cache, use_cache=True).logits[0, -1].float()
    t0 = time.time()
    for dia_id, line in transcript(conv):
        ids = tok(line, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        pos_of[dia_id] = line_start
        last = stream_ids(model, cache, runner, ids, nll, full, last)
        line_start += ids.shape[1]
    end = line_start
    secs = time.time() - t0

    records = []
    for qi, qa in enumerate(conv["qa"]):
        cat = qa.get("category")
        if cat == 5 or "answer" not in qa:
            continue
        gold = str(qa["answer"])
        ev = [pos_of[e] for e in qa.get("evidence", []) if e in pos_of]
        prompt = chat_turn(tok, QUESTION.format(q=qa["question"]), close_prev=True)
        gold_ids = tok(gold, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        rec = {"q": qa["question"], "gold": gold, "category": cat,
               "evidence_dist": [end - p for p in ev],
               # in the window at question time (sinks hold only the first n_sink tokens: the system prompt)
               "evidence_in_window": bool(ev) and all(end - p <= window for p in ev),
               "logp": answer_logprob(model, cache, prompt, gold_ids)}
        greedy = tok.decode(generate(model, cache, prompt, im_end, MAX_ANSWER, runner=runner))
        rec["greedy"], rec["f1_greedy"] = greedy, token_f1(greedy, gold)
        samples = []
        for s in range(seeds):
            g = torch.Generator(device="cuda").manual_seed(1000 * qi + s)
            ans = tok.decode(generate(model, cache, prompt, im_end, MAX_ANSWER, 0.7, g, runner=runner))
            samples.append({"answer": ans, "f1": token_f1(ans, gold)})
        rec["samples"] = samples
        records.append(rec)
    return {"tokens": end, "stream_seconds": round(secs, 1), "cache_len": cache_len(cache),
            "nll": nll, "qa": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--conversations", type=int, nargs="*", default=None, help="indices (default: all 10)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.config != "full" and not args.config.startswith("v2"):
        raise SystemExit("probes need checkpoints: use 'full' or a v2 config")
    model, tok = load_model(args.model)
    data = load_locomo()
    probe = make_cache(args.config, model)
    window = probe.window if isinstance(probe, SinkCacheV2) else 10 ** 9
    del probe
    out_dir = RESULTS / "dialogue" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in (args.conversations if args.conversations is not None else range(len(data))):
        path = out_dir / f"{args.config}_conv{i}.json"
        if path.exists():
            print("exists, skipping", path)
            continue
        try:
            res = run_conversation(model, tok, data[i], args.config, args.seeds, window)
        except torch.OutOfMemoryError:
            res = {"oom": True}
            torch.cuda.empty_cache()
        res.update({"config": args.config, "conversation": i, "sample_id": data[i].get("sample_id")})
        path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        if "qa" in res:
            f1 = sum(r["f1_greedy"] for r in res["qa"]) / max(1, len(res["qa"]))
            ppl = torch.tensor(res["nll"]).mean().exp().item()
            print(f"[{args.config}] conv {i}: {res['tokens']} tokens, ppl {ppl:.2f}, "
                  f"{len(res['qa'])} questions, greedy F1 {f1:.3f}", flush=True)
        else:
            print(f"[{args.config}] conv {i}: OOM", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
