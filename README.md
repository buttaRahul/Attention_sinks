# Attention sinks for Transformers 5

A StreamingLLM-style KV cache (keep the first few "sink" tokens plus a rolling
window, evict everything in between) written against the Transformers 5
per-layer cache API, with RoPE re-rotation on eviction.

Work in progress.
