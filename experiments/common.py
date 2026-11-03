"""Shared helpers: model loading, PG19 tokens, cache configs."""

import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from sinkcache import SinkCache
from sinkcache.decode_attention import register as register_decode_attention

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    # same weights as meta-llama/Llama-3.2-1B, without the gated-repo login
    "llama3.2-1b": "unsloth/Llama-3.2-1B",
}


def load_model(key: str, dtype=torch.bfloat16, attn: str = "sdpa_decode"):
    """attn='sdpa_decode': Triton flash-decode for one-token steps, SDPA for longer chunks
    (see sinkcache/decode_attention.py for why). Same backend for every cache config."""
    name = MODELS[key]
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


CONFIG_RE = re.compile(r"^(full|sink(\d+)_w(\d+)(_norot)?(_fp32k)?)$")


def make_cache(spec: str, model):
    """'full' -> DynamicCache (HF's default cache);
    'sink{n}_w{w}' -> SinkCache(n_sink=n, window=w), so 'sink0_w1024' is window-only;
    suffix '_norot' disables re-rotation (the position bug), '_fp32k' stores keys in fp32."""
    m = CONFIG_RE.match(spec)
    if not m:
        raise ValueError(f"bad cache spec {spec!r}")
    if spec == "full":
        return DynamicCache()
    return SinkCache.from_model(model, n_sink=int(m.group(2)), window=int(m.group(3)),
                                rerotate_keys=m.group(4) is None,
                                key_dtype=torch.float32 if m.group(5) else None)

