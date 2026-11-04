"""Shared helpers: model loading, PG19 tokens, cache configs."""

import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from sinkcache import SinkCache, SinkCacheV2
from sinkcache.decode_attention import register as register_decode_attention

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    # same weights as meta-llama/Llama-3.2-1B, without the gated-repo login
    "llama3.2-1b": "unsloth/Llama-3.2-1B",
}


def cap_to_physical_vram(margin_mib: int = 256) -> None:
    """Make PyTorch raise a real OOM at the card's physical VRAM.

    On Windows the NVIDIA driver's "sysmem fallback" silently pages VRAM out to system RAM
    once the card is full, so a growing cache never OOMs: it slows ~10x and eats host RAM.
    Capping the caching allocator at what is actually free keeps every measurement on-device."""
    free, total = torch.cuda.mem_get_info()
    cap = torch.cuda.memory_reserved() + free - margin_mib * 2**20
    torch.cuda.set_per_process_memory_fraction(cap / total)


def load_model(key: str, dtype=torch.bfloat16, attn: str = "sdpa_decode"):
    """attn='sdpa_decode': Triton flash-decode for one-token steps, SDPA for longer chunks
    (see sinkcache/decode_attention.py for why). Same backend for every cache config."""
    name = MODELS[key]
    cap_to_physical_vram()
    tok = AutoTokenizer.from_pretrained(name)
    if attn == "sdpa_decode":
        register_decode_attention()
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn)
    return model.to("cuda").eval(), tok


def kv_bytes_per_token(config, dtype_bytes: int = 2) -> int:
    """layers x 2 (K and V) x KV heads x head dim x bytes."""
    c = config.get_text_config(decoder=True)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return c.num_hidden_layers * 2 * c.num_key_value_heads * head_dim * dtype_bytes


def pg19_tokens(tokenizer, model_key: str, n_tokens: int) -> torch.Tensor:
    """PG19 test books, in dataset order, concatenated (separated by EOS) until n_tokens.
    Cached to data/ so every config sees exactly the same tokens."""
    DATA.mkdir(exist_ok=True)
    path = DATA / f"pg19_{model_key}_{n_tokens}.pt"
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset

    ds = load_dataset("emozilla/pg19-test", split="test")
    ids: list[int] = []
    for book in ds:
        ids += tokenizer(book["text"]).input_ids
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        if len(ids) >= n_tokens:
            break
    out = torch.tensor(ids[:n_tokens])[None]
    torch.save(out, path)
    return out


CONFIG_RE = re.compile(r"^(full|(v2)?sink(\d+)_w(\d+)(_norot)?(_bf16k)?(_pin(\d+))?)$")


def make_cache(spec: str, model):
    """'full' -> DynamicCache (HF's default cache);
    'sink{n}_w{w}' -> SinkCache(n_sink=n, window=w) (v1: window re-rotated each step, fp32 keys),
      so 'sink0_w1024' is window-only; suffix '_norot' disables re-rotation (the position bug),
      '_bf16k' stores keys in bf16 instead of the fp32 default (shows the accumulated drift);
    'v2sink{n}_w{w}' -> SinkCacheV2 (sinks moved forward, window untouched, bf16 keys);
      suffix '_pin{p}' gives it a budget of p pinned tokens (cache.pin(start, end))."""
    m = CONFIG_RE.match(spec)
    if not m:
        raise ValueError(f"bad cache spec {spec!r}")
    if spec == "full":
        return DynamicCache()
    n_sink, window = int(m.group(3)), int(m.group(4))
    if m.group(2):
        if m.group(5) or m.group(6):
            raise ValueError("_norot/_bf16k are v1 ablations")
        return SinkCacheV2.from_model(model, n_sink=n_sink, window=window,
                                      max_pinned=int(m.group(8) or 0))
    if m.group(7):
        raise ValueError("pinning needs a v2 cache")
    return SinkCache.from_model(model, n_sink=n_sink, window=window, rerotate_keys=m.group(5) is None,
                                key_dtype=None if m.group(6) else torch.float32)


def cache_len(cache) -> int:
    """Tokens a new query attends to (besides itself), for any cache type."""
    if isinstance(cache, SinkCacheV2):
        return cache.layers[0].in_cache_length()
    return cache.get_seq_length()


def tokens_seen(cache) -> int:
    return cache.tokens_seen if hasattr(cache, "tokens_seen") else cache.get_seq_length()


class Checkpoint:
    """Branch off a streamed conversation (ask a probe question) and come back to exactly where it was.
    DynamicCache: crop back to the saved length. SinkCacheV2: in-place snapshot/restore (buffer
    pointers kept, so a captured CUDA graph stays valid). v1 SinkCache: not supported here."""

    def __init__(self, cache):
        self.cache = cache
        if isinstance(cache, DynamicCache):
            self.state = cache.get_seq_length()
        elif isinstance(cache, SinkCacheV2):
            self.state = cache.snapshot()
        else:
            raise TypeError("probes need DynamicCache or SinkCacheV2")

    def restore(self):
        if isinstance(self.cache, DynamicCache):
            self.cache.crop(self.state)
        else:
            self.cache.restore(self.state)


def chat_turn(tok, user: str, system: str | None = None, reply: str | None = None, close_prev: bool = False):
    """Qwen3 chat-format token ids for one user turn (thinking disabled), optionally with the system
    prompt in front, the <|im_end|> closing the previous assistant reply, and a teacher-forced reply."""
    s = "<|im_end|>\n" if close_prev else ""
    if system is not None:
        s += f"<|im_start|>system\n{system}<|im_end|>\n"
    s += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if reply is not None:
        s += reply
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.cuda()


@torch.inference_mode()
def answer_logprob(model, cache, prompt_ids, answer_ids) -> float:
    """Mean log p(answer token | conversation, prompt) per token, teacher-forced, on a checkpoint."""
    ck = Checkpoint(cache)
    x = torch.cat([prompt_ids, answer_ids], 1)
    logits = model(x, past_key_values=cache, use_cache=True).logits[0].float()
    n = answer_ids.shape[1]
    lp = torch.log_softmax(logits[-n - 1:-1], -1).gather(1, answer_ids[0, :, None]).mean().item()
    ck.restore()
    return lp


@torch.inference_mode()
def generate(model, cache, prompt_ids, stop_id: int, max_new: int, temperature: float = 0.0,
             generator=None, runner=None) -> list[int]:
    """Decode a reply on a checkpoint (the cache is restored afterwards). temperature 0: greedy;
    otherwise Qwen3's non-thinking sampling (T, top-k 20, top-p 0.8). One-token steps go through
    `runner` (CUDA graph) when given."""
    ck = Checkpoint(cache)
    logits = model(prompt_ids, past_key_values=cache, use_cache=True).logits[0, -1]
    out = []
    for _ in range(max_new):
        if temperature == 0:
            nxt = logits.argmax()
        else:
            probs = torch.softmax(logits.float() / temperature, -1)
            top = torch.topk(probs, 20)
            p = top.values / top.values.sum()
            p = torch.where(torch.cumsum(p, 0) - p < 0.8, p, 0)
            nxt = top.indices[torch.multinomial(p / p.sum(), 1, generator=generator)[0]]
        nid = int(nxt)
        if nid == stop_id:
            break
        out.append(nid)
        step = runner.step if runner is not None else (
            lambda x: model(x, past_key_values=cache, use_cache=True).logits)
        logits = step(nxt.view(1, 1))[0, -1]
    ck.restore()
    return out


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token F1 (lower-cased, punctuation and articles stripped): LoCoMo's QA metric."""
    import collections
    import string

    def norm(s):
        s = "".join(c for c in str(s).lower() if c not in string.punctuation)
        return [w for w in s.split() if w not in ("a", "an", "the", "and")]
    p, g = norm(pred), norm(gold)
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not p or not g or common == 0:
        return float(p == g)
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)
