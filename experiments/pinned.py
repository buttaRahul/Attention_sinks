"""Experiment 8: pinning the system prompt, and pinning facts, at a fixed cache budget.

A voice agent's system prompt (rules, caller profile) is said once, at the
start, and must hold for the whole call. With 4 sinks + a 1,020-token window
it is evicted after ~1K tokens. Three ways to spend the same 1,024-token budget:

  sinks4         v2, 4 sinks + 1,020 window (plain StreamingLLM)
  sysprompt      v2, n_sink = the whole system prompt (S tokens) + (1,024 - S) window
  sysprompt+pin  as above, plus up to 128 pinned tokens for user turns that state a fact
                 (a trivial detector: the turn contains a digit), window shrunk to match
  full           no eviction (reference)

The call is a scripted support call per seed: randomized agent name, caller
name, account number, address and phone number, and a filler of ordinary
questions. Facts in the system prompt: agent name, account number. Facts said
mid-call by the caller: new address (turn 8), callback number (turn 14).
One rule: end every reply with a fixed sentence.

The transcript (user turns and assistant replies) is generated once per seed
with the full cache and then teacher-forced for every config, so every config
sees exactly the same tokens. At probe turns the conversation is
checkpointed, a probe question is asked, the cache is restored:

  - recall probes: teacher-forced mean log p of the gold answer, plus greedy and
    `--samples` sampled answers checked for containing it;
  - instruction probe: a normal question; greedy and sampled replies checked
    for the closing sentence.

    python experiments/pinned.py --build --seeds 0 1 2 3 4     # fixed transcripts (full cache)
    python experiments/pinned.py --seeds 0 1 2 3 4               # all configs
"""

import argparse
import json
import random
import re
import sys

import torch

from common import (DATA, RESULTS, answer_logprob, chat_turn, generate, load_model, make_cache, tokens_seen)
from sinkcache import SinkCacheV2, StreamRunner

BUDGET, PIN_BUDGET = 1024, 128
TURNS = 64
PROBE_TURNS = [6, 16, 32, 48, 64]
REPLY_TOKENS = 120
CLOSER = "Is there anything else I can help you with today?"

AGENTS = ["Ava", "Milo", "Priya", "Jonas", "Leila", "Tomas", "Nadia", "Oscar"]
CALLERS = ["Daniel Okafor", "Maria Lindqvist", "Hiro Tanaka", "Grace Mbeki", "Luis Romero", "Anna Novak"]
STREETS = ["Maple Avenue", "Harbour Road", "Elm Street", "Kingfisher Lane", "Station Road", "Birch Close"]
FILLER = [
    "Can you explain how my monthly bill is calculated?", "Why did my bill go up this month?",
    "What's the difference between the Basic and the Plus plan?", "Do you charge for calls abroad?",
    "How do I set up voicemail on my phone?", "My internet keeps dropping in the evening. Any idea why?",
    "Is there a cheaper plan for someone who mostly uses Wi-Fi?", "How long does a SIM swap take?",
    "Can I pause my contract while I travel?", "What happens if I go over my data allowance?",
    "How do I turn on Wi-Fi calling?", "Can you tell me what roaming costs in Europe?",
    "Is my phone eligible for an upgrade?", "How do I block spam calls?",
    "What's your policy on early termination?", "Can I add a second line for my daughter?",
    "Do you offer paper bills or only email?", "How do I check how much data I've used?",
    "My router's light is blinking orange. What does that mean?", "Can I keep my number if I switch plans?",
    "Is there a discount for paying a year in advance?", "How do I reset my voicemail PIN?",
    "Why is my phone showing 'SOS only'?", "Do you have 5G coverage in rural areas?",
    "Can I get a refund for the outage last week?", "How do I transfer my contacts to a new phone?",
    "What's included in the device insurance?", "How do I change the name on the account?",
    "Can I set a spending cap on my account?", "How do I get an itemised bill?",
    "Why are my texts arriving late?", "Is there a family plan?", "How secure is my account login?",
    "Can you walk me through setting up autopay?", "What does the 'network access fee' on my bill mean?",
    "How do I report a lost phone?", "Can I use my own router?", "How fast is the Plus plan's broadband?",
    "Do unused minutes roll over?", "How do I unlock my phone for another network?",
    "What are your opening hours for the store?", "Is there a student discount?",
    "How do I cancel the international add-on?", "Can I get a temporary data boost?",
    "Why was I charged twice this month?", "How do I update my email address on file?",
    "What happens to my plan if I move abroad?", "Can you explain the late payment fee?",
    "How do I enable parental controls?", "Is there an app to manage my account?",
    "Can you send a technician to check my line?", "What's the warranty on my router?",
    "How do I switch off data roaming?", "Can I change my billing date?",
    "Do you support eSIM?", "What's the best plan for working from home?",
    "How do I file a complaint?", "Can I get a copy of my contract?",
]
PROBES = {
    "account": "Sorry, can you read my account number back to me? Just the number.",
    "agent": "Sorry, I missed it earlier: what's your name?",
    "address": "Can you confirm the new address I gave you? Just the address.",
    "phone": "What callback number did I give you? Just the number.",
    "rule": "One more question: how do I check my data usage?",
}


def scenario(seed: int) -> dict:
    r = random.Random(seed)
    s = {"agent": r.choice(AGENTS), "caller": r.choice(CALLERS),
         "account": "".join(r.choice("0123456789") for _ in range(8)),
         "address": f"{r.randint(12, 980)} {r.choice(STREETS)}",
         "phone": f"07{r.randint(100, 999)} {r.randint(100000, 999999)}"}
    s["system"] = (
        f"You are {s['agent']}, a phone support agent for Northwind Mobile, speaking with a caller on a voice "
        "call. Your replies are read aloud by a speech synthesiser, so keep them to two or three short "
        "sentences, never use lists, markdown, or emoji, and spell out anything that is hard to hear.\n\n"
        "Rules you must follow for the whole call:\n"
        f"1. End every single reply with exactly this sentence: \"{CLOSER}\"\n"
        f"2. If the caller asks your name, tell them it is {s['agent']}.\n"
        "3. Never promise refunds or credits; say a supervisor will review the request.\n"
        "4. If the caller gives new details (address, phone number), confirm them back once.\n\n"
        "Caller profile (verified at the start of the call):\n"
        f"- Name: {s['caller']}\n"
        f"- Account number: {s['account']}\n"
        "- Plan: Plus, 30 GB data, unlimited calls and texts, contract until next March\n"
        "- Payment: autopay by direct debit on the 3rd of each month\n"
        "- Devices: one handset, one home router\n\n"
        "Product notes: Basic is 10 GB for 12 pounds a month; Plus is 30 GB for 18 pounds; Max is "
        "unlimited for 25 pounds. Roaming in Europe is included on Plus and Max. Wi-Fi calling and eSIM "
        "are supported on all current handsets. Early termination costs the remaining months at half price."
    )
    filler = FILLER[:]
    r.shuffle(filler)
    turns = [f"Hi, this is {s['caller'].split()[0]}. I have a question about my account. " + filler[0]]
    fi = 1
    for t in range(2, TURNS + 1):
        if t == 8:
            turns.append(f"Oh, before I forget: I moved last month. My new address is {s['address']}.")
        elif t == 14:
            turns.append(f"If we get cut off, please call me back on {s['phone']}.")
        else:
            turns.append(filler[fi % len(filler)])
            fi += 1
    s["user_turns"] = turns
    return s


def path_for(seed):
    return DATA / f"pinned_call_seed{seed}.json"


@torch.inference_mode()
def build(model, tok, seed):
    """Generate the fixed transcript for one seed with a full cache (sampled replies)."""
    s = scenario(seed)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cache = make_cache("full", model)
    g = torch.Generator(device="cuda").manual_seed(seed)
    replies = []
    for i, user in enumerate(s["user_turns"]):
        prompt = chat_turn(tok, user, system=s["system"] if i == 0 else None, close_prev=i > 0)
        # generate() restores the cache; then the turn and its reply are fed for real
        reply_ids = generate(model, cache, prompt, im_end, REPLY_TOKENS, 0.7, g)
        reply = tok.decode(reply_ids)
        replies.append(reply)
        model(torch.cat([prompt, torch.tensor([reply_ids], device="cuda")], 1) if reply_ids else prompt,
              past_key_values=cache, use_cache=True)
        print(f"seed {seed} turn {i + 1}: {len(reply_ids)} tokens, seen {tokens_seen(cache)}", flush=True)
    s["replies"] = replies
    path_for(seed).write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def has_fact(text: str, fact: str) -> bool:
    norm = lambda x: re.sub(r"[^0-9a-z]", "", x.lower())
    return norm(fact) in norm(text)


def follows_rule(text: str) -> bool:
    norm = lambda x: re.sub(r"[^a-z]", "", x.lower())
    return norm(text).endswith(norm(CLOSER))


def configs_for(sys_len: int) -> dict:
    return {
        "full": "full",
        "sinks4": "v2sink4_w1020",
        "sysprompt": f"v2sink{sys_len}_w{BUDGET - sys_len}",
        "sysprompt+pin": f"v2sink{sys_len}_w{BUDGET - sys_len - PIN_BUDGET}_pin{PIN_BUDGET}",
    }


@torch.inference_mode()
def evaluate(model, tok, s, name, spec, samples):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cache = make_cache(spec, model)
    runner = StreamRunner(model, cache, use_graph=True) if isinstance(cache, SinkCacheV2) else None
    pin = name.endswith("+pin")
    facts = {"account": s["account"], "agent": s["agent"], "address": s["address"], "phone": s["phone"]}
    out = []
    for i, (user, reply) in enumerate(zip(s["user_turns"], s["replies"])):
        turn = i + 1
        if turn in PROBE_TURNS:
            for kind, q in PROBES.items():
                if kind == "address" and turn <= 8 or kind == "phone" and turn <= 14:
                    continue
                prompt = chat_turn(tok, q, close_prev=True)
                rec = {"turn": turn, "probe": kind, "tokens_seen": tokens_seen(cache)}
                if kind != "rule":
                    gold = tok(facts[kind], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                    rec["logp"] = answer_logprob(model, cache, prompt, gold)
                answers = [tok.decode(generate(model, cache, prompt, im_end, 80, runner=runner))]
                for k in range(samples):
                    g = torch.Generator(device="cuda").manual_seed(10_000 * turn + k)
                    answers.append(tok.decode(generate(model, cache, prompt, im_end, 80, 0.7, g, runner=runner)))
                ok = [follows_rule(a) if kind == "rule" else has_fact(a, facts[kind]) for a in answers]
                rec.update({"greedy": answers[0], "greedy_ok": ok[0], "sampled_ok": ok[1:]})
                out.append(rec)
        # feed the turn for real (teacher-forced): user part (pinned if it states a fact), then the reply
        user_ids = chat_turn(tok, user, system=s["system"] if i == 0 else None, close_prev=i > 0)
        start = tokens_seen(cache)
        model(user_ids, past_key_values=cache, use_cache=True)
        if pin and re.search(r"\d", user):
            cache.pin(start, tokens_seen(cache))
        reply_ids = tok(reply, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        if reply_ids.shape[1]:
            model(reply_ids, past_key_values=cache, use_cache=True)
    region = cache.region_size if isinstance(cache, SinkCacheV2) else None
    return {"config": name, "spec": spec, "tokens": tokens_seen(cache), "region_size": region, "probes": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    model, tok = load_model(args.model)
    if args.build:
        for seed in args.seeds:
            build(model, tok, seed)
        return
    out_dir = RESULTS / "pinned" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        s = json.loads(path_for(seed).read_text(encoding="utf-8"))
        # n_sink = every token of the first chunk up to the end of the system message
        sys_len = tok(f"<|im_start|>system\n{s['system']}<|im_end|>\n", add_special_tokens=False,
                      return_tensors="pt").input_ids.shape[1]
        for name, spec in configs_for(sys_len).items():
            if args.configs and name not in args.configs:
                continue
            path = out_dir / f"seed{seed}_{name.replace('+', '_')}.json"
            if path.exists():
                continue
            res = evaluate(model, tok, s, name, spec, args.samples)
            res.update({"seed": seed, "system_tokens": sys_len})
            path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            late = [p for p in res["probes"] if p["turn"] >= 32]
            acc = sum(p["greedy_ok"] for p in late) / max(1, len(late))
            print(f"seed {seed} {name}: {res['tokens']} tokens, late-probe greedy accuracy {acc:.2f}", flush=True)


if __name__ == "__main__":
    main()
