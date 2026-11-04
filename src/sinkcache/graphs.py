"""One-token streaming with an optional CUDA graph for the steady state.

v1 (`SinkCache`): with in-cache positions, once the cache is full every decode
step is the same computation: same shapes, same position (n_sink + window),
same buffers. One captured step is replayed for every later token.

v2 (`SinkCacheV2`): shapes and buffers are fixed too, but the position grows.
The slot index and the sinks' rotation come from a device-side counter inside
the cache, and position_ids are fed from a static device tensor, so one
captured step still serves every later token. A step that has to move a
pinned token into the region runs eagerly, and a change of region size
(attention length) triggers a recapture.

A DynamicCache grows every step and cannot be captured this way.

On this Windows laptop eager decoding is CPU-bound (~45 us per kernel launch,
~500 launches per step), so this changes per-token time from ~90 ms to a few ms.
It does not change the math: replay runs the same kernels on the same buffers
(tested bit-for-bit against eager).
"""

import torch

from .cache import SinkCache
from .cache_v2 import SinkCacheV2


class StreamRunner:
    def __init__(self, model, cache, use_graph: bool = True):
        self.model, self.cache = model, cache
        self.v2 = isinstance(cache, SinkCacheV2)
        self.use_graph = use_graph and isinstance(cache, (SinkCache, SinkCacheV2)) and torch.cuda.is_available()
        self.graph = None
        self.static_ids = self.static_logits = self.static_pos = None
        self._key = None
        self.replays = 0

    @torch.no_grad()
    def step(self, ids: torch.Tensor) -> torch.Tensor:
        """Feed ids of shape [batch, q]; returns logits [batch, q, vocab]. One-token steps on a
        steady cache are graph-replayed; anything else runs eagerly (and may change the cache
        layout, which invalidates the graph: it is then recaptured).
        When graphed, the returned tensor is overwritten by the next step: consume it first."""
        one = ids.shape[-1] == 1
        if self.graph is not None and self._layout() != self._key:
            self.graph = None  # buffers reallocated or region resized: recapture
        if self.graph is not None and one and self.cache.is_steady:
            self.static_ids.copy_(ids)
            if self.v2:
                self.static_pos.fill_(self.cache.tokens_seen)
            self.graph.replay()
            self._advance()
            self.replays += 1
            return self.static_logits
        if self.use_graph and one and self.cache.is_steady:
            self._capture(ids)
            self.graph.replay()  # capture records but does not execute: run this step now
            return self.static_logits
        return self.model(ids, past_key_values=self.cache, use_cache=True).logits

    def _advance(self):
        """Python-side counters don't advance inside a replay."""
        if self.v2:
            self.cache.advance_after_replay()
        else:
            for layer in self.cache.layers:
                layer.tokens_seen += 1
                layer.tokens_evicted += 1

    def _layout(self):
        ptrs = [(l._kbuf.data_ptr(), l._vbuf.data_ptr()) for l in self.cache.layers]
        return ptrs, (self.cache.layout_key() if self.v2 else None)

    def _capture(self, ids):
        self._key = self._layout()
        self.static_ids = ids.clone()
        kwargs = {}
        if self.v2:
            self.static_pos = torch.full((ids.shape[0], 1), self.cache.tokens_seen, dtype=torch.long,
                                         device=ids.device)
            kwargs["position_ids"] = self.static_pos
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.static_logits = self.model(self.static_ids, past_key_values=self.cache, use_cache=True,
                                            **kwargs).logits
