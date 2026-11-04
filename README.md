# Attention sinks for Transformers 5

A StreamingLLM KV cache written against the **Transformers 5 per-layer cache
API**. It keeps the first few "sink" tokens forever, keeps a rolling window of
recent tokens, and evicts everything in between. Positions are handled by
**re-rotating cached keys on eviction**. `SinkCache` was removed from core
Transformers, so this re-implements it as a `CacheLayerMixin` subclass that
plugs into any full-attention RoPE model without patching the model.

```python
from sinkcache import SinkCache

cache = SinkCache.from_model(model, n_sink=4, window=1020)
out = model(input_ids, past_key_values=cache, use_cache=True)   # stream forever
```

**Headline (Qwen3-0.6B, bf16, RTX 4060 Laptop 8 GB, PG19 streamed one token at a time):**
over the 40K tokens where the full cache is inside Qwen3's trained position range, a
1,024-token sink cache is **+6.3% in perplexity** (95% CI +4.7% to +8.2%) against the full cache,
and the gap grows with distance (+5.5% at 1–8K, +10.0% at 32–40K). A window-only cache of
the same size is **14.3× worse**. Memory is constant (the full cache grows **112 KiB/token**
and runs out at **50,176 tokens**). Measured the same way, in GPU kernel time per token, the
sink cache stays at **22.5–25.1 ms** from 1K to 32K while the full cache goes **20.8 → 62.2 ms**.

![Perplexity vs position](results/ppl_vs_position.png)

---

## Results

### Experiment 1: streaming perplexity

PG19 test books, concatenated in dataset order (book 1 ends at token 57,722).
Each token is fed through the cache one at a time, and the plot shows the mean NLL of the next
token per 1,024-token bucket. The three main configs all hold **1,024 tokens** of cache.

| range (tokens) | full cache | 4 sinks + 1,020 window | sinks vs full | 95% CI | window only (1,024) | window vs full |
|---|---:|---:|---:|---:|---:|---:|
| **1,024–40,960 (full cache in its trained range)** | 20.78 | 22.09 | **+6.3%** | +4.7 to +8.2% | 296.8 | **14.3×** |
| 1,024–8,192 | 21.96 | 23.16 | +5.5% | +2.6 to +9.2% | 293.4 | 13.4× |
| 8,192–16,384 | 22.61 | 23.11 | +2.2% | −0.5 to +5.6% | 327.3 | 14.5× |
| 16,384–32,768 | 19.83 | 21.21 | +7.0% | +5.1 to +9.0% | 279.8 | 14.1× |
| 32,768–40,960 | 19.95 | 21.95 | +10.0% | +5.4 to +14.7% | 306.0 | 15.3× |
| 40,960–50,176 (full cache past its trained range) | 26.65 | 26.89 | +0.9% | −5.2 to +9.4% | 393.9 | 14.8× |

CI: paired bootstrap over 1,024-token blocks. The text is one book, so neighbouring tokens are not
independent samples.

- The window-only cache is identical to the others until token 1,024. From the first
  eviction, perplexity jumps from ~20 to ~290 and stays there.
- **The gap grows with distance.** The full cache keeps getting more out of long-range context as
  the book goes on (characters, plot), which a 1,020-token window cannot see. The trend is noisy
  (the 8–16K bin's CI includes zero), but 32–40K is clearly worse than the start.
- **Past 40,960 tokens the comparison is unfair to the full cache.** That is Qwen3's
  `max_position_embeddings`, and the full cache degrades beyond it, so the gap there (+0.9%) mostly
  measures the full cache breaking. Averaging over all 50K tokens would dilute the headline to +5.3%.
  The sink cache never goes past position 1,024, so position range never limits it.
- The step up at ~58K is the start of PG19 book 2, while the cache still holds book 1's last
  pages. The full cache had already run out of memory by then, so there is no baseline for that range.

**Ablations** (perplexity over tokens 2K–16K; the full cache scores 22.44 there):

![Ablations](results/ablations.png)

| sinks (window 1,020) | ppl | | cache size | 4 sinks + window | window only |
|---:|---:|---|---:|---:|---:|
| 0 | 312.96 | | 256 | 25.55 | 613.16 |
| 1 | 23.34 | | 512 | 24.17 | 430.43 |
| 2 | 23.35 | | 1,024 | 23.29 | 312.69 |
| 4 | 23.29 | | 2,048 | 23.24 | 259.08 |
| 8 | 23.40 | | | | |

- **One sink token recovers all of the gap** (313 → 23.3). 2, 4 and 8 are the same to
  within noise. StreamingLLM found that 1 sink recovers most of the gap and 4 is enough; for Qwen3,
  1 is already enough.
- A 256-token sink cache (25.6) beats a 2,048-token window-only cache (259) by 10×.
  Cache size matters much less than keeping the sink.
- **Position handling matters:** without re-rotation the same cache scores **47,642**.

**Second model, Llama-3.2-1B** (`unsloth/Llama-3.2-1B`, the same weights as
the gated Meta repo; it uses `llama3` RoPE scaling, which `find_inv_freq` picks up). This run only
went to **16K tokens**, so it checks that the result transfers, not how the gap behaves at long range:

| Llama-3.2-1B, first 16K tokens only | 0–1K | 1K–2K | 2K–16K |
|---|---:|---:|---:|
| full cache | 9.23 | 9.48 | 9.73 |
| window only (1,024) | 9.23 | 132.60 | 177.28 |
| 4 sinks + 1,020 | 9.23 | 9.52 | **10.45 (+7.4%)** |

### Experiment 2: memory against length

![Memory](results/memory_vs_tokens.png)

| tokens | full cache | sinks / window (1,024) |
|---:|---:|---:|
| 1,024 | 1,258 MiB | 1,370 MiB |
| 16,384 | 2,938 MiB | 1,378 MiB |
| 32,768 | 4,730 MiB | 1,378 MiB |
| 49,152 | 6,522 MiB | 1,378 MiB |
| 50,176 | **OOM** | 1,378 MiB (to 65,536 and beyond) |

The full cache measures **112.00 KiB/token**, exactly the figure derived from the config
(see [bytes per token](#bytes-per-token)), and runs out of memory at 50,176 tokens on the 8 GB card.

The sink cache holds 1,025 slots × 28 layers of fp32 K and bf16 V, which is **168 MiB**
(see [precision](#the-precision-cost-of-re-rotation) for why the keys are fp32). With bf16
keys it would be 112 MiB. The rest of its 241 MiB above the weights is the CUDA-graph memory pool
and logits.

### Experiment 3: decode latency against length

![Latency](results/latency_vs_length.png)

Per-token decode at each context length. Three rounds, config order rotated each round,
medians reported. Two metrics, which are **not** comparable with each other:

- **GPU time**: the sum of CUDA kernel durations per token (CUPTI), both configs run eagerly.
  This is the like-for-like comparison: same execution mode, same attention kernel. Broken down
  by kernel type below.
- **CUDA-graph wall clock**: only the sink cache can be graph-replayed (see the caveats below),
  so this number is a separate point about what a fixed-shape cache allows, not a head-to-head.

| config | context | GPU ms/token | range | linear | attention | cache copies | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| full cache | 1,024 | 20.8 | 20.1–23.9 | 10.4 | 1.7 | 3.1 | 5.5 |
| full cache | 8,192 | 27.5 | 27.5–28.9 | 8.0 | 5.0 | 10.6 | 3.9 |
| full cache | 16,384 | 36.8 | 36.8–40.6 | 7.0 | 10.2 | 15.8 | 3.9 |
| full cache | 32,768 | **62.2** | 61.6–62.5 | 7.0 | 18.3 | **33.1** | 3.9 |
| 4 sinks + 1,020 | 1,024 | 22.5 | 19.4–28.2 | 9.9 | 1.6 | 5.9 | 5.2 |
| 4 sinks + 1,020 | 32,768 | **25.1** | 23.6–27.5 | 11.2 | 1.9 | 6.3 | 5.7 |

On GPU time the sink cache is flat (22.5–27.8 ms at every length, run-to-run noise from laptop
clocks) and the full cache rises 3×. At 1K the sink cache is *slower*: ~6 ms of its step is the
"cache copies" column (the eviction kernel re-rotating 1,020 fp32 keys per layer, the value shift,
and the fp32→bf16 key copy attention reads). [v2](#v2-move-the-sinks-not-the-window) is designed to
remove it (not measured yet).

**Does it match the memory-bound prediction?** Yes, once you count the bytes HF actually moves.
Decode reads every weight (1.19 GB) and the whole KV cache once per token. The brief's
estimate of "weights + KV" gives about 3–4× slower at 32K. But HF's `DynamicCache` appends with
`torch.cat`, which **reads and rewrites the entire cache every step**, so the per-token traffic
is **weights + 3 × KV**:

| at 32K tokens (KV = 3.76 GB) | bytes moved | measured | effective rate |
|---|---:|---:|---:|
| `torch.cat` cache copy (read + write) | 7.5 GB | 33.1 ms | 227 GB/s (copy ceiling) |
| attention reads K and V once | 3.76 GB | 18.3 ms | 205 GB/s |
| linear layers (weights) | 1.19 GB | 7.0 ms | 170 GB/s |

Predicted slope: 3 × 112 KiB ÷ 220 GB/s = **1.56 µs per extra token of context**.
Measured: (62.2 − 20.8) ms ÷ 31.7K tokens = **1.31 µs/token**. So at 32K the full cache costs
3× its short-context time, and **over half of it is the cache copy, not attention**. An
in-place cache, like the preallocated buffer used here for sinks, would remove that.

**Wall-clock caveats.** On this Windows laptop eager decoding is *launch-bound*:
~45 µs per kernel launch × ~500 launches is a 60–100 ms/token floor, which hides most of the KV
growth (right panel). The GPU also drops its clocks while waiting on the CPU (780 MHz SM
observed), which makes every kernel slower in eager mode and adds noise to the summed kernel
times. A sink cache has a property that sidesteps this: once full, every decode step has the same
shapes *and the same positions*, because the new token is always at position `n_sink + window`.
So the whole step can be captured in **one CUDA graph** and replayed, which gives **15.3–15.7 ms/token
wall clock at every length**. A growing `DynamicCache` can't be captured that way. This is a
separate result from the table above: it compares a graph-replayed path with nothing, and it says
what a fixed-shape cache makes possible, not that sinks are 4× faster than a full cache.

### Experiment 4: a long conversation, and what sinks don't fix (illustration)

**This is an illustration, not a measurement:** one sampled run per config, and because replies
are sampled, the configs saw different amounts of text (4,654 / 5,668 / 2,035 tokens). [Experiments 6 and 8](#experiments-68-does-it-hold-for-a-voice-agent-code-ready-not-run-yet) are
the controlled versions (not run yet).

50 turns of Qwen3 chat (thinking off), streamed turn by turn. Turn 1 plants a fact:
*"my order ID is 4417"*. It is probed at turn 3, while the fact is still in the window, and at turn 50,
after it has been evicted from the sink and window caches. Each probe records the greedy answer and
the teacher-forced log-probability of "4417" after "Your order ID is ".

![Chat](results/chat.png)

| config | tokens seen | turn 3 answer | log p | turn 50 answer | log p | memory, turn 1 → 50 | distinct-2 |
|---|---:|---|---:|---|---:|---|---:|
| full cache | 4,654 | `4417` | −0.00 | `4417` | **−0.01** | 1,160 → 1,655 MiB | 0.94 |
| 4 sinks + 1,020 | 5,668 | `4417` | −0.00 | `123456789` | **−13.7** | 1,321 → 1,375 MiB | 0.90 |
| window only | 2,035 | `4417` | −0.00 | *(empty)* | −29.9 | 1,321 → 1,386 MiB | 0.20 |

- **Sinks give stable fluency, not memory.** The sink cache keeps answering normally at
  flat memory, but once the fact is evicted it is gone: the model guesses a number, and "4417"
  is e^−13.7 ≈ 1-in-900,000. That is the expected behaviour, not a bug.
  Evicted content can only come back by re-retrieving it (RAG, summarisation, or a larger cache).
- The window-only cache degenerates. From about turn 20 every reply is an immediate end-of-turn
  token (distinct-2 of 0.20).

### Experiments 6–8: does it hold for a voice agent? (code ready, not run yet)

The PG19 results are about books. A voice agent has different requirements: calls of a few thousand
tokens, many concurrent sessions per GPU, real dialogue, and a system prompt that must hold for the
whole call. Three experiments target this. The code is written but **none has been run yet, so
there are no numbers here.**

#### Experiment 6: real dialogue

[`experiments/dialogue.py`](experiments/dialogue.py), on [LoCoMo](https://github.com/snap-research/locomo):
10 long multi-session conversations with ~1,500 annotated questions (single-hop, multi-hop, temporal,
open-domain), each with evidence turns. For full, sinks (v2) and window-only it measures:

- perplexity on the dialogue, by position;
- per question: the teacher-forced log p of the gold answer, plus greedy and 3 sampled answers
  scored by LoCoMo's token F1;
- how far back the evidence was, split by whether it is still in the window.

Uncertainty comes from bootstrapping over conversations, not from a single run.

#### Experiment 7: the crossover point

[`experiments/crossover.py`](experiments/crossover.py) runs at session lengths of 2K, 4K, 8K and 16K. Assuming ~200 transcript tokens per minute plus
a ~1K-token system prompt, those correspond to roughly 5-, 15-, 30- and 60-minute calls. It measures:

- KV memory per session;
- the largest number of concurrent sessions that fit in 8 GB;
- aggregate decode throughput at that batch.

Per-session GPU time against length (512 to 32K, full / v1 / v2) comes from `latency.py`.

#### Experiment 8: pinned system prompt and pinned facts

[`experiments/pinned.py`](experiments/pinned.py) runs a scripted 64-turn support call per seed (5 seeds,
randomized agent name, account number, address and phone number). The transcript is generated once and
teacher-forced, so every config sees the same tokens. Probes run at turns 6–64:

- recall of system-prompt facts and of facts the caller states mid-call;
- compliance with a rule from the system prompt ("end every reply with …").

Configs, all at a 1,024-token budget: 4 sinks; `n_sink` = the whole system prompt; the system prompt
plus 128 pinned tokens for caller turns that state a fact; and the full cache as reference.

---

## Implementation

[`src/sinkcache/cache.py`](src/sinkcache/cache.py) contains `SinkLayer(CacheLayerMixin)` and
`SinkCache(Cache)`.

### Positions: the bug everyone makes

To the model, the kept tokens must look like a contiguous sequence 0…cache_len−1.
Transformers 5 derives positions from the cache:
`position_ids = cache.get_seq_length() + arange(q_len)`. So `SinkLayer.get_seq_length()`
returns the **in-cache** length, and new tokens land right after the kept ones. What remains is
to keep the kept keys consistent with that.

**(a) Re-rotate on eviction (this repo, and HF's removed `SinkCache`).** Keys are cached
*after* RoPE. RoPE rotates each pair of dimensions by `pos·θᵢ`, and rotations compose by adding
angles, so a key rotated at position p and then rotated by −Δ is *exactly* the key rotated at
p−Δ. When `drop` tokens are evicted from the front of the window, every surviving window key
moves `drop` positions earlier, so it is rotated by −drop. Sinks never move. Values carry no
position, so they are just shifted.

Qwen3 applies `k_norm`, a per-channel RMSNorm, *before* RoPE. So the cache holds
`rope(norm(k))`, and the identity still holds. That is tested, not assumed:
`test_qwen3_keys_shift` runs the real model with every position shifted by +3,000 and checks that the
re-rotated keys match the unshifted run at all 28 layers. `test_rotation_order_matters` shows the
identity would *fail* if the norm came after RoPE.

**(b) Cache keys before RoPE (StreamingLLM's approach).** Store un-rotated keys and apply RoPE
at attention time using in-cache positions 0…L−1. There is no re-rotation and no accumulated
rounding, and eviction is a pure slice. The costs:

- Every step rotates *all* cached keys, which is O(cache) extra work per token, though it fuses
  naturally into the attention kernel.
- It needs a change to the model's attention code, because HF applies RoPE before handing keys to
  the cache.

(a) works through the public cache API with no model patching.

### When eviction happens

`update()` runs *before* attention and must return the keys this step attends to. The current
queries were rotated at positions L…L+q−1, so they must see the old keys at their old positions.
Eviction therefore happens *after* attention: it is deferred to the start of the next `update`.
Between steps the logical cache is `n_sink + window` tokens; a decode step attends over
`n_sink + window + 1`. HF's own `DynamicSlidingWindowLayer` makes the same choice.

### The precision cost of re-rotation

This is the finding I didn't expect. Every window key is re-rotated **once per step** for its whole
life in the window, and in bf16 each re-rotation rounds the stored key again. The rounding error
random-walks, roughly √n × 2⁻⁹ after n steps:

- **In the unit tests:** 1.3% relative key error after 60 re-rotations in bf16, against
  9×10⁻⁷ in fp32.
- **In perplexity:** with bf16 key storage, a *bigger* sink cache got *worse*, the opposite of
  what it should do.

| cache size (4 sinks) | 256 | 512 | 1,024 | 2,048 |
|---|---:|---:|---:|---:|
| bf16 keys | 25.46 | 23.66 | **31.81** | **62.02** |
| fp32 keys (default) | 25.55 | 24.17 | 23.29 | 23.24 |

So `SinkCache` stores keys in **fp32** by default. Attention reads a bf16 copy; values stay
bf16. This costs +50% KV memory (168 MiB vs 112 MiB for 1,024 tokens). bf16 storage is still
available (`key_dtype=None`, the `_bf16k` configs).

That workaround treats the symptom. The cause is re-rotating 1,020 keys every step at all, which v2 removes.

### v2: move the sinks, not the window

[`src/sinkcache/cache_v2.py`](src/sinkcache/cache_v2.py), `SinkCacheV2`. The obvious question about
v1 is "why rotate 1,020 keys per step instead of 4?". RoPE attention only depends on relative
position, so the same attention pattern can be had the other way round:

- **Window keys are never touched.** They stay exactly as the model wrote them, rotated at their
  absolute stream positions, in bf16.
- **The query sits at its absolute position.** `get_seq_length()` returns tokens seen, so
  Transformers builds `position_ids` from it. `get_query_offset` / `get_mask_sizes` report the
  in-cache size for the mask. This is the same split HF's own `StaticSlidingWindowLayer` uses.
- **Only the sinks move.** Each step they are rotated *from their stored original keys* to sit
  directly before the window: one rotation, one rounding, no accumulation.
- **Eviction is O(1).** The window is a ring buffer of `window + 1` slots, and a new token overwrites
  the slot of the one that just left. There is no shift, no re-rotation, no fp32→bf16 copy. Slot order
  doesn't matter to attention because every key carries its position in its rotation.

| | v1 (`SinkCache`) | v2 (`SinkCacheV2`) |
|---|---|---|
| keys rotated per step | 1,020 per layer | 4 per layer |
| roundings per window key | one per step it lives (√n drift in bf16) | none: stored once, never modified |
| key storage | fp32 (168 MiB for 1,024 tokens) | bf16 (112 MiB) |
| eviction | shift the whole window | overwrite one ring slot |
| position range | bounded (≤ cache size) | grows with the stream (fp64 angles for the sinks) |

Relative distances are identical to v1, so logits match v1 to 1×10⁻³ in fp32 (`test_matches_v1`).
The tests in [`tests/test_cache_v2.py`](tests/test_cache_v2.py) cover:

- an independent reference: pre-RoPE keys recorded by hooks and rotated once by HF's code at the
  positions v2 claims;
- bf16 window keys bit-for-bit what the model wrote, after 236 evictions;
- chunks longer than the window;
- CUDA-graph replay bitwise equal to eager, including pin moves that force eager steps and recapture
  (written; its last run hit a token-count bug in the test itself, fixed, not yet re-run).

The trade-off: positions grow without bound. HF computes `position × θ` in fp32, so at very large
positions (≫100K) the query and key angles lose precision. v1 never leaves position 1,024.

The `v2` ablation (perplexity, memory, GPU time per step against v1's fp32 and bf16 keys) is set up in
`run_ppl.sh v2` and `latency.py`, but **not run yet**.

**Pinning.** v2's sink slots generalise to a *region*: sinks, plus tokens the application pins with
`cache.pin(start, end)`. A pinned token stays in the window like any other, and when it would be
evicted it moves into the region instead. The region is re-placed before the window each step, so
pinned tokens look to the model as if they were said just before the window, after the sinks. This
is what [Experiment 8](#experiment-8-pinned-system-prompt-and-pinned-facts) uses.

### Making it fast on this machine

Three pieces exist because the obvious PyTorch path was slow here. Each is tested against a reference.

- **Preallocated in-place buffer.** K/V live in `n_sink + window + 1` slots updated in place,
  instead of a `torch.cat` of the whole cache every step.
- **Fused Triton eviction** ([`kernels.py`](src/sinkcache/kernels.py)). It shifts the window left
  and re-rotates keys in one pass: one read and one write per element, with fp32 math in registers.
  It works in place because each program walks rows from low to high with a barrier between a
  chunk's loads and its stores. The unfused PyTorch version moved about as many bytes per step as
  the model weights. Values come out bit-identical to the PyTorch path; keys agree to within one rounding.
- **Split-K flash-decoding attention backend** ([`decode_attention.py`](src/sinkcache/decode_attention.py)),
  registered as `attn_implementation="sdpa_decode"`. On the Windows PyTorch build no SDPA backend
  reads a growing GQA cache once:
  - Flash attention isn't compiled in.
  - Memory-efficient attention has no GQA support.
  - cuDNN re-plans for every new length, which took 0.5 s/token.
  - HF's default dispatch lands on the math backend, which upcasts to fp32 and takes 18.8 ms per layer at 32K.

  The kernel handles a whole GQA group per program, reaches 127 GB/s, and is used for *every*
  config. Its logits are as close to an fp32 reference as stock SDPA's (mean |Δ| 0.0605 vs 0.0609).

One trap worth knowing: on Windows the NVIDIA driver's *sysmem fallback* pages VRAM out to system
RAM when the card is full. A growing cache then never hits OOM; it slows ~10× and eats host RAM.
The experiments cap PyTorch's allocator at the free VRAM so that a real OOM is raised.

---

## Bytes per token

From the model config (`kv_bytes_per_token` in [`experiments/common.py`](experiments/common.py)):

| | layers | KV heads | head dim | K+V bytes per token (bf16) |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 8 | 128 | 28 × 2 × 8 × 128 × 2 = **114,688 B = 112 KiB** |
| Llama-3.2-1B | 16 | 8 | 64 | 16 × 2 × 8 × 64 × 2 = **32,768 B = 32 KiB** |

A 32K-token Qwen3-0.6B cache is 3.5 GiB, about 3× the model's 1.19 GB of weights. A 1,024-token
cache is 112 MiB whatever the stream length. Llama-3.2-1B has twice the weights but grows its cache
3.5× more slowly, because it has fewer layers and half the head dimension. GQA and head size, not
parameter count, decide how fast the KV cache grows.

## Why the first tokens become sinks

Softmax attention must sum to 1. A head with nothing useful to attend to still has to put its
probability mass *somewhere*. During training, the only tokens every query can see are the first
few (because of causal masking), so the model learns to use them as a dump: their keys draw large
attention scores whatever their content, and their values are close to neutral.

Evict them and that mass is redistributed over whatever is left in the window. That changes every
head's output at every layer, so the model is running on activations it never saw in training and
perplexity explodes (313 here). Keeping even **one** of them restores the distribution the model
expects (23.3). Their content doesn't matter, only that they exist and are visible to every query.

## Where it doesn't help

- **Memory of evicted content.** Sinks give stable fluency over unbounded streams, not
  long-term recall (Experiment 4). Anything outside the window is gone.
- **Quality versus a full cache.** Sinks are 6.3% worse in perplexity (in the full cache's trained
  range), and the gap grows with distance, because the model only ever sees the last 1,020 tokens.
  It is a fixed-budget trade-off, not free context.
- **Prefill and chunked input.** A chunk longer than the cache gets full causal attention within
  itself, then gets trimmed. Within a chunk this is not streaming attention.
- **Batching, beam search, rollback.** `crop` and `reorder_cache` are not supported, and the CUDA
  graph path assumes batch-1 decode.
- **Speed at short context.** At 1K tokens the sink cache is no faster than a full cache: the eviction
  kernel, fp32→bf16 key copy and value shift cost ~6 ms of GPU time per step (the "cache copies"
  column). It only wins once the full cache is several thousand tokens long.

---

## Testing

`pytest` runs 63 tests in about 6 minutes:

- **Before any eviction**, logits are bitwise identical to `DynamicCache`.
- **Rotation**: rope(k, p) re-rotated by −Δ equals rope(k, p−Δ), at fp32 tolerance that scales
  with position. Our rotation matches HF's `apply_rotary_pos_emb` bit for bit. The Qwen3
  whole-network shift test and the norm-after-RoPE negative control are described
  [above](#positions-the-bug-everyone-makes).
- **After eviction (the strongest test)**: hooks record each layer's pre-RoPE keys and values
  during the sink run. The kept tokens' keys are then rotated **directly** at their new positions
  0…cap−1 by HF's own rotary code and put into a `DynamicCache`. The next-token logits from that
  cache must match the sink cache.
  - In fp32 they agree to 1×10⁻⁵ after 236 evictions; keys agree to 9×10⁻⁷ relative and values
    bit for bit.
  - Checked for 4 sinks and for window-only (0 sinks), through both the Triton and PyTorch paths.
  - A literal fresh forward pass over only the kept tokens can't match: those tokens' K/V were
    computed when they could still see tokens that were later evicted. This test isolates exactly
    what the cache claims, which is attention over the kept K/V at contiguous positions.
- **Teeth**: with re-rotation disabled, the same check fails by more than 1 logit.
- **Precision**: fp32 key storage cuts the accumulated drift from 1.3% to one bf16 rounding.
- **Kernels**: Triton eviction vs PyTorch (values bit-exact), flash-decode vs an fp32 reference across
  GQA shapes and lengths from 1 to 32K, and the registered backend in the real model.
- **CUDA graphs**: replay is bitwise identical to eager, including when multi-token chunks
  reallocate the buffers between graphed steps.

## Reproduce

```bash
uv venv --python 3.11 .venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[dev]"          # transformers 5.22.1, triton / triton-windows

pytest                                # correctness, ~6 min
bash experiments/run_ppl.sh all       # Experiments 1-2 + ablations + Llama, ~4 h on a 4060 Laptop
python experiments/latency.py         # Experiment 3, ~20 min
python experiments/chat.py --config full          # Experiment 4, ~8 min per config
python experiments/chat.py --config sink4_w1020
python experiments/chat.py --config sink0_w1024
python experiments/plot.py            # plots + results/tables.md
```

Setup: RTX 4060 Laptop (8 GB, 256 GB/s), torch 2.11 + CUDA 12.8, transformers 5.22.1,
triton-windows 3.9, Windows 11, bf16 weights. Full tables: [`results/tables.md`](results/tables.md).

## Layout

```
src/sinkcache/
  cache.py              v1: SinkLayer / SinkCache (eviction, re-rotation, in-place buffer)
  cache_v2.py           v2: SinkCacheV2 (sinks moved forward, ring-buffer window, pinning)
  rope.py               RoPE helpers, reference re-rotation
  kernels.py            fused in-place shift + re-rotation (Triton)
  decode_attention.py   split-K flash-decode GQA attention backend (Triton)
  graphs.py             StreamRunner: CUDA-graph replay of steady-state decode
tests/                  pytest suite
experiments/            ppl.py, latency.py, chat.py, dialogue.py, crossover.py, pinned.py, plot.py, run_ppl.sh
results/                CSVs, JSON, plots, tables.md
```
