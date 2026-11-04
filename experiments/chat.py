"""Experiment 4: a long multi-turn chat streamed through the cache, and what sinks don't fix.

A 50-turn conversation (Qwen3 chat format, thinking disabled) is streamed
turn by turn: each user turn is fed as one chunk, the reply is decoded token
by token. Per turn we record cache length, allocated memory, per-token decode
time, and simple degeneration metrics of the reply.

Turn 1 plants a fact ("my order ID is 4417"). It is probed twice:
  turn 3  (the fact is still inside the window for every config)
  turn 50 (the fact was evicted long ago from the sink and window caches)
At each probe we record the model's actual (greedy) answer, and the
teacher-forced log-probability of the digits "4417" after "Your order ID is"
measured on a throwaway copy of the cache.

    python experiments/chat.py --config sink4_w1020
"""

import argparse
import copy
import json
import statistics
import sys
import time

import torch

from common import RESULTS, cache_len, load_model, make_cache, tokens_seen

SYSTEM = "You are a helpful assistant. Keep every answer to one short paragraph."
FACT = "Hi! Before we start: my order ID is 4417, please keep it in mind. First question: how does a lighthouse lens work?"
PROBE = "Quick check: what is my order ID? Reply with just the number."
FILLER = [
    "Why is the sky blue?", "Give me a tip for keeping basil alive indoors.",
    "What is the difference between weather and climate?", "Explain what a hash table is.",
    "Suggest a name for a grey cat.", "How do bees make honey?",
    "What's a good way to start learning the guitar?", "Why do we have leap years?",
    "Explain compound interest with a small example.", "What causes tides?",
    "Give me a short recipe for tomato soup.", "What is photosynthesis?",
    "How do noise-cancelling headphones work?", "What's the tallest mountain in Africa?",
    "Write a two-line poem about rain.", "How does a refrigerator keep food cold?",
    "What is the Pythagorean theorem?", "Recommend a habit for better sleep.",
    "Why do onions make you cry?", "Explain what DNS does on the internet.",
    "How do airplanes stay in the air?", "What is a black hole?",
    "Give me three words that rhyme with 'light'.", "How does a vaccine train the immune system?",
    "What is the difference between a virus and a bacterium?", "How do I make cold brew coffee?",
    "Explain recursion to a ten-year-old.", "Why is the ocean salty?",
    "What were the main causes of the French Revolution?", "How do solar panels make electricity?",
    "What is inflation?", "Suggest a fun weekend activity for a rainy day.",
    "How does GPS find your location?", "What is the speed of light, roughly?",
    "Why do cats purr?", "Explain the rules of chess pawns briefly.",
    "How are rainbows formed?", "What is a prime number? Give examples.",
    "How do I remove a coffee stain from a shirt?", "Explain what an API is.",
    "Why do leaves change colour in autumn?", "What's the difference between RAM and storage?",
    "How do volcanoes form?", "Tell me an interesting fact about octopuses.",
    "What is the water cycle?",
]
MAX_NEW = 160


def user_chunk(tok, text, first=False):
    s = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n" if first else "\n")
    s += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.cuda()


def distinct2(ids):
    pairs = list(zip(ids, ids[1:]))
    return len(set(pairs)) / max(1, len(pairs))


@torch.inference_mode()
def answer_logprob(model, tok, cache, prompt_ids, answer_prefix, answer):
    """log p(answer | conversation, prompt, answer_prefix), on a copy of the cache."""
    c = copy.deepcopy(cache)
    pre = tok(answer_prefix, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    ans = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    x = torch.cat([prompt_ids, pre, ans], 1)
    logits = model(x, past_key_values=c, use_cache=True).logits[0].float()
    n = ans.shape[1]
    lp = torch.log_softmax(logits[-n - 1:-1], -1).gather(1, ans[0, :, None]).sum().item()
    del c
    torch.cuda.empty_cache()
    return lp


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turns", type=int, default=50)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    probe_turns = (3, args.turns)

    model, tok = load_model(args.model)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    cache = make_cache(args.config, model)
    turns, pending = [], None
    fillers = iter(FILLER * 2)
    for turn in range(1, args.turns + 1):
        is_probe = turn in probe_turns
        text = FACT if turn == 1 else PROBE if is_probe else next(fillers)
        chunk = user_chunk(tok, text, first=turn == 1)
        if pending is not None:  # the previous reply's final <|im_end|> has not been fed yet
            chunk = torch.cat([pending, chunk], 1)
        rec = {"turn": turn, "user": text}
        if is_probe:
            rec["logp_4417"] = answer_logprob(model, tok, cache, chunk, "Your order ID is ", "4417")

        logits = model(chunk, past_key_values=cache, use_cache=True).logits[0, -1]
        out, times = [], []
        for _ in range(MAX_NEW):
            if is_probe:  # deterministic answer for the recall check
                nxt = logits.argmax()
            else:  # Qwen3's recommended non-thinking sampling
                probs = torch.softmax(logits.float() / 0.7, -1)
                top = torch.topk(probs, 20)
                p = top.values / top.values.sum()
                keep = torch.cumsum(p, 0) - p < 0.8
                p = torch.where(keep, p, 0)
                nxt = top.indices[torch.multinomial(p / p.sum(), 1, generator=gen)[0]]
            nid = nxt.item()
            out.append(nid)
            if nid == im_end:
                break
            torch.cuda.synchronize()
            t = time.perf_counter()
            logits = model(nxt.view(1, 1), past_key_values=cache, use_cache=True).logits[0, -1]
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1e3)
        ended = out[-1] == im_end
        pending = torch.tensor([[im_end]], device="cuda") if ended else None
        reply_ids = out[:-1] if ended else out
        rec.update({
            "reply": tok.decode(reply_ids),
            "reply_tokens": len(reply_ids), "ended": ended, "distinct2": round(distinct2(reply_ids), 3),
            "tokens_seen": tokens_seen(cache),
            "cache_len": cache_len(cache),
            "alloc_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            "decode_ms_median": round(statistics.median(times), 2) if times else None,
        })
        if is_probe:
            rec["recalled"] = "4417" in rec["reply"]
        turns.append(rec)
        print(f"[{args.config}] turn {turn:2d} seen={rec['tokens_seen']:5d} cache={rec['cache_len']:5d} "
              f"mem={rec['alloc_mib']:7.1f}MiB ms={rec['decode_ms_median']} "
              + (f"logp4417={rec['logp_4417']:.2f} reply={rec['reply']!r}" if is_probe else
                 f"len={rec['reply_tokens']} d2={rec['distinct2']}"), flush=True)

    path = RESULTS / "chat" / f"{args.model}_{args.config}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": args.config, "model": args.model, "turns": turns}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
