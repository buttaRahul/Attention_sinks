"""One-token streaming with an optional CUDA graph for the steady state.

With in-cache positions, once a SinkCache is full every decode step is the same
computation: same shapes, same position (n_sink + window), same buffers. So one
captured step can be replayed for every later token. A DynamicCache grows every
step and cannot be captured this way.

On this Windows laptop eager decoding is CPU-bound (~45 us per kernel launch,
~500 launches per step), so this changes per-token time from ~90 ms to a few ms.
It does not change the math: replay runs the same kernels on the same buffers
(tested bit-for-bit against eager).
"""

import torch

from .cache import SinkCache


class StreamRunner:
    def __init__(self, model, cache, use_graph: bool = True):
        self.model, self.cache = model, cache
        self.use_graph = use_graph and isinstance(cache, SinkCache) and torch.cuda.is_available()
        self.graph = None
        self.static_ids = self.static_logits = None

    @torch.no_grad()
    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """Feed ids of shape [batch, 1]; returns logits [batch, 1, vocab].
        When graphed, the returned tensor is overwritten by the next step: consume it first."""
        if self.graph is not None:
            self.static_ids.copy_(ids)
            self.graph.replay()
            for layer in self.cache.layers:  # Python-side counters don't advance inside a replay
                layer.tokens_seen += 1
                layer.tokens_evicted += 1
            return self.static_logits
        if self.use_graph and ids.shape[-1] == 1 and self.cache.is_steady:
            self._capture(ids)
            self.graph.replay()  # capture records but does not execute: run this step now
            return self.static_logits
        return self.model(ids, past_key_values=self.cache, use_cache=True).logits

    def _capture(self, ids):
        self.static_ids = ids.clone()
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.static_logits = self.model(self.static_ids, past_key_values=self.cache, use_cache=True).logits
