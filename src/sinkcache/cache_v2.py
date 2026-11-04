"""v2: move the sinks forward instead of moving the window back.

v1 (`SinkCache`) keeps every kept token at in-cache positions 0..L-1. When a
token is evicted, all 1,020 window keys shift one position earlier, so every
window key is re-rotated every step. That costs a pass over the whole key cache
per step, and in bf16 the repeated rounding drifts (hence v1's fp32 keys).

RoPE attention only depends on *relative* position. So the same attention
pattern can be had the other way round:

  * window keys stay exactly as the model wrote them, rotated at their absolute
    stream positions, and are never touched again;
  * the query sits at its absolute position t (Transformers takes it from
    `get_seq_length()`, which here returns the number of tokens seen);
  * only the few sink tokens move: each step they are rotated, from their
    stored original keys, to sit immediately before the window.

Relative distances are identical to v1 (query→window key, query→sink j), so
the logits are the same up to rounding. Per step this rotates 4 keys instead of
1,020, rounds each sink key once (from its original), and never rounds a
window key at all, so bf16 storage is exact.

Storage
-------
The window lives in a ring buffer of `window + 1` slots: the token at absolute
position p >= n_sink sits in slot (p - n_sink) mod (window + 1). A decode step
writes the new token over the slot of the token that left the window one step
ago (eviction is O(1): no shift, no copy). Attention does not care about slot
order, because every key already carries its position in its rotation.

After the ring come the "region" slots: the sinks, plus any pinned tokens
(see below), rotated to positions ws-R..ws-1 where ws is the window's first
position and R the region size. Attention reads one contiguous view:
ring + region.

Pinning
-------
`cache.pin(start, end)` marks absolute positions. A pinned token is kept in
the window like any other; when it would be evicted it moves into the region
instead (its key is still at its original rotation, so it becomes its own
"original"). The region is placed contiguously before the window, so pinned
tokens look to the model as if they were said just before the window, after
the sinks, in their original order. The pinned budget is `max_pinned`; past
it the oldest pinned token is dropped.

Steady-state decode (one token, cache full) uses only device tensors for the
slot index and the rotation offset, so it can be captured in a CUDA graph
(`StreamRunner` feeds position_ids from a device tensor). Everything else
(prefill, chunks, warm-up, pin moves) runs a general path with host ints.
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .cache import find_inv_freq
from .rope import rotate_half


class SinkLayerV2(CacheLayerMixin):
    is_sliding = False
    is_croppable = False

    def __init__(self, n_sink: int, window: int, inv_freq: torch.Tensor, max_pinned: int = 0,
                 pinned: set[int] | None = None):
        super().__init__()
        if n_sink < 0 or window < 1 or max_pinned < 0:
            raise ValueError(f"need n_sink >= 0, window >= 1, max_pinned >= 0; got {n_sink}, {window}, {max_pinned}")
        self.n_sink, self.window, self.max_pinned = n_sink, window, max_pinned
        self.ring = window + 1
        self.region_cap = n_sink + max_pinned
        self.inv_freq = inv_freq
        self.pinned = pinned if pinned is not None else set()
        self.t = 0                       # absolute position of the next token = tokens seen
        self.region_pos: list[int] = []  # original positions of region tokens (sinks, then pinned)
        self.moved_upto = 0              # pinned positions below this have been handled
        self.pins_dropped = 0

    # ------------------------------------------------------------------ storage

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        b, h, _, d = key_states.shape
        n = self.ring + self.region_cap
        self._kbuf = torch.zeros(b, h, n, d, dtype=self.dtype, device=self.device)
        self._vbuf = torch.zeros(b, h, n, d, dtype=self.dtype, device=self.device)
        self._reg_k = torch.zeros(b, h, max(self.region_cap, 1), d, dtype=self.dtype, device=self.device)
        self._reg_pos = torch.zeros(max(self.region_cap, 1), dtype=torch.long, device=self.device)
        self._t_dev = torch.zeros((), dtype=torch.long, device=self.device)
        self._arange = torch.arange(max(self.region_cap, 1), device=self.device)
        self.inv_freq = self.inv_freq.to(self.device, torch.float64)
        self.is_initialized = True

    def _slot(self, p: int) -> int:
        return (p - self.n_sink) % self.ring

    def _window_start(self, t: int) -> int:
        return max(min(t, self.n_sink), t - self.window)

    # ------------------------------------------------------------------ rotation

    def _rotate_region(self, ws) -> torch.Tensor:
        """Region keys rotated from their originals to positions ws-R..ws-1 (ws: int or 0-d tensor).
        Angles in fp64 (positions grow without bound), math in fp32, one rounding to the model dtype."""
        R = len(self.region_pos)
        delta = ws - R + self._arange[:R] - self._reg_pos[:R]
        freqs = delta.to(torch.float64)[:, None] * self.inv_freq
        emb = torch.cat((freqs, freqs), -1)
        cos, sin = emb.cos().float(), emb.sin().float()
        x = self._reg_k[:, :, :R].float()
        rot = cos.shape[-1]
        if rot == x.shape[-1]:
            out = x * cos + rotate_half(x) * sin
        else:
            xr = x[..., :rot]
            out = torch.cat((xr * cos + rotate_half(xr) * sin, x[..., rot:]), -1)
        return out.to(self.dtype)

    # ------------------------------------------------------------------ region bookkeeping

    def _region_append(self, k: torch.Tensor, v: torch.Tensor, p: int):
        """k, v: [b, h, 1, d]; k at its original rotation (position p)."""
        if len(self.region_pos) == self.region_cap:
            if self.max_pinned == 0:
                raise RuntimeError("region full")
            self._region_drop_oldest_pin()
        j = len(self.region_pos)
        self._reg_k[:, :, j:j + 1].copy_(k)
        self._vbuf[:, :, self.ring + j:self.ring + j + 1].copy_(v)
        self._reg_pos[j] = p
        self.region_pos.append(p)

    def _region_drop_oldest_pin(self):
        s, R, r = self.n_sink, len(self.region_pos), self.ring
        self._reg_k[:, :, s:R - 1] = self._reg_k[:, :, s + 1:R].clone()
        self._vbuf[:, :, r + s:r + R - 1] = self._vbuf[:, :, r + s + 1:r + R].clone()
        self._reg_pos[s:R - 1] = self._reg_pos[s + 1:R].clone()
        del self.region_pos[s]
        self.pins_dropped += 1

    def _move_pins(self, bound: int, chunk=None):
        """Pinned positions in [moved_upto, bound) have left the window: move them into the region.
        Each is either still in its ring slot or (chunk=(k, v, t0)) in the chunk being stored."""
        if self.pinned and bound > self.moved_upto:
            for p in sorted(p for p in self.pinned if self.moved_upto <= p < bound and p >= self.n_sink):
                if chunk is not None and p >= chunk[2]:
                    i = p - chunk[2]
                    self._region_append(chunk[0][:, :, i:i + 1], chunk[1][:, :, i:i + 1], p)
                else:
                    sl = self._slot(p)
                    self._region_append(self._kbuf[:, :, sl:sl + 1], self._vbuf[:, :, sl:sl + 1], p)
        self.moved_upto = max(self.moved_upto, bound)

    def next_step_moves_pin(self) -> bool:
        """Would a one-token step at the current position need a host-side pin move afterwards?"""
        return (self.t - self.window) in self.pinned and self.t - self.window >= self.n_sink

    # ------------------------------------------------------------------ cache API

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        q = key_states.shape[-2]
        if q == 1 and self.t >= self.n_sink + self.window:
            return self._decode_step(key_states, value_states)
        return self._general_step(key_states, value_states)

    def _decode_step(self, k, v):
        """Steady state: device tensors only (CUDA-graph safe). The new token overwrites the slot of
        the token that left the window last step; the region is re-placed before the new window."""
        R = len(self.region_pos)
        slot = ((self._t_dev - self.n_sink) % self.ring).view(1)
        self._kbuf.index_copy_(2, slot, k)
        self._vbuf.index_copy_(2, slot, v)
        if R:
            self._kbuf[:, :, self.ring:self.ring + R].copy_(self._rotate_region(self._t_dev - self.window))
        self._t_dev.add_(1)
        self.t += 1
        self._move_pins(self.t - self.window)   # host side; never needed inside a captured step
        n = self.ring + R
        self.keys, self.values = self._kbuf[:, :, :n], self._vbuf[:, :, :n]
        return self.keys, self.values

    def _general_step(self, k, v):
        """Any chunk, warm-up, or a step that moves a pin. Attention gets a gathered copy
        [region (rotated), window in position order, new chunk]; then the new tokens are stored."""
        t, q = self.t, k.shape[-2]
        ws = self._window_start(t)
        R = len(self.region_pos)
        win = range(max(ws, self.n_sink), t)
        slots = torch.tensor([self._slot(p) for p in win], dtype=torch.long, device=self.device)
        r = self.ring
        keys = torch.cat((self._rotate_region(ws), self._kbuf.index_select(2, slots), k), -2)
        values = torch.cat((self._vbuf[:, :, r:r + R], self._vbuf.index_select(2, slots), v), -2)

        # store: sinks first (positions < n_sink), then pins leaving the window, then the ring
        for i in range(q):
            if t + i < self.n_sink:
                self._region_append(k[:, :, i:i + 1], v[:, :, i:i + 1], t + i)
        self._move_pins(t + q - self.window, chunk=(k, v, t))
        lo = max(self.n_sink, t + q - self.ring, t)
        if lo < t + q:
            idx = torch.tensor([self._slot(p) for p in range(lo, t + q)], dtype=torch.long, device=self.device)
            self._kbuf.index_copy_(2, idx, k[:, :, lo - t:])
            self._vbuf.index_copy_(2, idx, v[:, :, lo - t:])
        self.t = t + q
        self._t_dev.fill_(self.t)
        self.keys, self.values = keys, values
        return keys, values

    def get_seq_length(self) -> int:
        """Absolute position of the next token (Transformers builds position_ids from this)."""
        return self.t

    def in_cache_length(self) -> int:
        """Tokens a new query attends to besides itself: region + current window."""
        return len(self.region_pos) + (self.t - max(self._window_start(self.t), self.n_sink)
                                       if self.t > self.n_sink else 0)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.in_cache_length() + query_length, 0

    def get_max_length(self) -> int:
        return self.region_cap + self.window

    def kept(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Logical cache for the next query: keys/values in order [region, window by position], and
        the positions they are rotated at (region: re-placed before the window; window: absolute)."""
        t = self.t
        ws = self._window_start(t)
        R = len(self.region_pos)
        win = list(range(max(ws, self.n_sink), t))
        slots = torch.tensor([self._slot(p) for p in win], dtype=torch.long, device=self.device)
        r = self.ring
        k = torch.cat((self._rotate_region(ws), self._kbuf.index_select(2, slots)), -2)
        v = torch.cat((self._vbuf[:, :, r:r + R], self._vbuf.index_select(2, slots)), -2)
        return k, v, [ws - R + j for j in range(R)] + win

    def reset(self) -> None:
        self.t = self.moved_upto = self.pins_dropped = 0
        self.region_pos = []
        self.keys = self.values = None
        self.is_initialized = False

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError("evicted tokens are gone; use snapshot()/restore() on the cache")

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("beam search is not supported")


class SinkCacheV2(Cache):
    """Attention sinks with the sinks moved forward and the window left alone (see module docstring).

        cache = SinkCacheV2.from_model(model, n_sink=4, window=1020)
        out = model(input_ids, past_key_values=cache, use_cache=True)

    `max_pinned`: extra region slots for pinned tokens (`cache.pin(start, end)`).
    """

    def __init__(self, n_sink: int, window: int, inv_freq: torch.Tensor, num_layers: int, max_pinned: int = 0):
        self.pinned: set[int] = set()
        layers = [SinkLayerV2(n_sink, window, inv_freq, max_pinned, self.pinned) for _ in range(num_layers)]
        super().__init__(layers=layers)
        self.n_sink, self.window, self.max_pinned = n_sink, window, max_pinned

    @classmethod
    def from_model(cls, model, n_sink: int = 4, window: int = 1020, **kwargs) -> "SinkCacheV2":
        config = model.config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
        if any(t != "full_attention" for t in layer_types):
            raise ValueError(f"SinkCacheV2 supports full-attention models only, got {set(layer_types)}")
        return cls(n_sink, window, find_inv_freq(model), config.num_hidden_layers, **kwargs)

    # positions come from get_seq_length (absolute); the mask is over what the layer returns
    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].in_cache_length()

    def pin(self, start: int, end: int) -> None:
        """Pin absolute positions [start, end). Must be called before they leave the window."""
        if self.max_pinned == 0:
            raise ValueError("cache was built with max_pinned=0")
        if start < self.layers[0].moved_upto:
            raise ValueError(f"positions before {self.layers[0].moved_upto} have already been evicted")
        self.pinned.update(range(start, end))

    @property
    def tokens_seen(self) -> int:
        return self.layers[0].t

    @property
    def tokens_evicted(self) -> int:
        layer = self.layers[0]
        return layer.t - layer.in_cache_length()

    @property
    def region_size(self) -> int:
        return len(self.layers[0].region_pos)

    @property
    def is_steady(self) -> bool:
        """The next one-token step takes the device-only path and moves no pin: safe to capture/replay."""
        layer = self.layers[0]
        return (layer.is_initialized and layer.t >= self.n_sink + self.window
                and not layer.next_step_moves_pin())

    def layout_key(self):
        """Anything that changes the captured computation: region size (attention length)."""
        return self.region_size

    def advance_after_replay(self):
        for layer in self.layers:
            layer.t += 1
            layer.moved_upto = max(layer.moved_upto, layer.t - layer.window)

    def snapshot(self):
        """Device copies of all state; restore() writes them back in place (buffer pointers kept,
        so a captured CUDA graph stays valid)."""
        return [(l._kbuf.clone(), l._vbuf.clone(), l._reg_k.clone(), l._reg_pos.clone(), l.t,
                 list(l.region_pos), l.moved_upto) for l in self.layers], set(self.pinned)

    def restore(self, snap):
        layers, pinned = snap
        for l, (kb, vb, rk, rp, t, rpos, mu) in zip(self.layers, layers):
            l._kbuf.copy_(kb), l._vbuf.copy_(vb), l._reg_k.copy_(rk), l._reg_pos.copy_(rp)
            l.t, l.region_pos, l.moved_upto = t, list(rpos), mu
            l._t_dev.fill_(t)
        self.pinned.clear()
        self.pinned.update(pinned)
