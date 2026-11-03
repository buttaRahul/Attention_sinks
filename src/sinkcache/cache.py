"""Attention-sink KV cache for the Transformers 5 per-layer cache API.

Each layer keeps the first `n_sink` tokens forever plus a rolling window of the
most recent `window` tokens; everything in between is evicted.

Positions
---------
Transformers assigns positions from the cache: a new chunk gets positions
`get_seq_length() + arange(q_len)`. This layer reports its *in-cache* length,
so a new token is always placed right after the kept tokens. For that to be
consistent, the kept keys must sit at contiguous positions 0..cache_len-1.

Keys are cached *after* RoPE (that is what the model hands to `update`). When
`drop` tokens are evicted from the front of the window, every surviving window
key moves `drop` positions earlier, so it is rotated by -drop. RoPE rotations
compose by adding angles, so this is exactly the key rotated at its new
position. Sinks never move. Values carry no position and are just shifted.

When eviction happens
---------------------
`update` must return the keys that *this* step's attention will use, and
attention runs after `update` returns. The current queries were rotated at
positions cache_len..cache_len+q_len-1, so they must see the old keys at their
old positions plus the new keys. Eviction therefore has to come after
attention. It is deferred to the start of the next `update` ("pending"
tokens): append, attend over everything, and at the next call evict down to
`n_sink + window` and re-rotate before appending again.

So between steps the logical cache is `n_sink + window` tokens (what
`get_seq_length` reports and the next position is based on), and a decode
step attends over `n_sink + window + 1` keys. The same accounting holds for
every configuration compared here.

Storage
-------
K/V live in a preallocated buffer of `n_sink + window + 1` slots and are
updated in place: no per-step `torch.cat` of the whole cache. Once the cache
is full, every decode step does exactly the same work at exactly the same
positions (the new token is always at position n_sink + window), so the step
can be captured in a CUDA graph and replayed (see experiments/common.py).
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .rope import rope_cos_sin

try:
    from .kernels import shift_window_
except ImportError:  # no triton: the PyTorch path is used
    shift_window_ = None


class SinkLayer(CacheLayerMixin):
    """One layer's cache: `n_sink` sink tokens + a rolling window of `window` tokens."""

    is_sliding = False      # masks treat it as a full-attention layer over what it returns
    is_croppable = False    # evicted tokens are gone; there is nothing to roll back to

    def __init__(self, n_sink: int, window: int, inv_freq: torch.Tensor, rerotate_keys: bool = True,
                 use_triton: bool | None = None):
        super().__init__()
        if n_sink < 0 or window < 1:
            raise ValueError(f"need n_sink >= 0 and window >= 1, got {n_sink}, {window}")
        self.n_sink = n_sink
        self.window = window
        self.capacity = n_sink + window
        self.inv_freq = inv_freq
        self.rerotate_keys = rerotate_keys
        self.length = 0                 # valid slots in the buffer, including pending evictions
        self.tokens_seen = 0
        self.tokens_evicted = 0
        self.use_triton = use_triton    # None: use the fused kernel when it applies
        self._rot: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._kbuf = self._vbuf = None

    # ------------------------------------------------------------------ storage

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.inv_freq = self.inv_freq.to(self.device)
        self._alloc(key_states, self.capacity + 1)
        self._rotation(1)  # the steady-state shift; built now so a graph capture never builds it
        fused_ok = (shift_window_ is not None and self.device.type == "cuda"
                    and 2 * self.inv_freq.numel() == key_states.shape[-1])  # full rotary only
        if self.use_triton and not fused_ok:
            raise RuntimeError("use_triton=True but the fused kernel does not apply here")
        self.use_triton = fused_ok if self.use_triton is None else self.use_triton
        self.is_initialized = True

    def _alloc(self, like, slots):
        b, h, _, d = like.shape
        k = torch.empty(b, h, slots, d, dtype=self.dtype, device=self.device)
        v = torch.empty(b, h, slots, d, dtype=self.dtype, device=self.device)
        if self.length:
            k[:, :, :self.length] = self._kbuf[:, :, :self.length]
            v[:, :, :self.length] = self._vbuf[:, :, :self.length]
        self._kbuf, self._vbuf = k, v

    # ------------------------------------------------------------------ rotation

    def _rotation(self, delta: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 (cos, sin, sin with the first half negated) for a rotation by -delta."""
        if delta not in self._rot:
            cos, sin = rope_cos_sin(torch.tensor(-delta, device=self.device), self.inv_freq)
            half = sin.shape[-1] // 2
            self._rot[delta] = (cos, sin, torch.cat((-sin[:half], sin[half:])))
        return self._rot[delta]

    def _rerotate(self, keys: torch.Tensor, delta: int) -> torch.Tensor:
        """keys rotated by -delta, in fp32 (returned in fp32; the caller rounds once on store).
        Same math as sinkcache.rope.rerotate, with cos/sin cached per delta and rotate_half
        written as a roll so it is a handful of kernels."""
        cos, _, sin_signed = self._rotation(delta)
        rot_dim = cos.shape[-1]
        x = keys.float()
        if rot_dim == x.shape[-1]:
            return x.mul(cos).addcmul_(x.roll(rot_dim // 2, dims=-1), sin_signed)
        xr = x[..., :rot_dim]
        out = xr.mul(cos).addcmul_(xr.roll(rot_dim // 2, dims=-1), sin_signed)
        return torch.cat((out, x[..., rot_dim:]), dim=-1)

    def _evict_pending(self):
        drop = self.length - self.capacity
        if drop <= 0:
            return
        s, cap, n = self.n_sink, self.capacity, self.length
        if self.use_triton:
            cos, sin, _ = self._rotation(drop) if self.rerotate_keys else (None, None, None)
            shift_window_(self._kbuf, s, drop, self.window, cos, sin)
            shift_window_(self._vbuf, s, drop, self.window)
        else:
            src_k = self._kbuf[:, :, s + drop:n]
            new_k = self._rerotate(src_k, drop) if self.rerotate_keys else src_k.clone()
            new_v = self._vbuf[:, :, s + drop:n].clone()  # overlapping ranges: copy via a temporary
            self._kbuf[:, :, s:cap].copy_(new_k)
            self._vbuf[:, :, s:cap].copy_(new_v)
        self.length = cap
        self.tokens_evicted += drop
        if self._kbuf.shape[-2] > cap + 1:  # a long chunk grew the buffer: shrink back
            self._alloc(self._kbuf, cap + 1)

    # ------------------------------------------------------------------ cache API

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self._evict_pending()
        q = key_states.shape[-2]
        end = self.length + q
        if end > self._kbuf.shape[-2]:
            self._alloc(key_states, end)
        self._kbuf[:, :, self.length:end].copy_(key_states)
        self._vbuf[:, :, self.length:end].copy_(value_states)
        self.length = end
        self.tokens_seen += q
        self.keys, self.values = self._kbuf[:, :, :end], self._vbuf[:, :, :end]
        # This step's attention sees everything, at positions that match the queries.
        return self.keys, self.values

    def get_seq_length(self) -> int:
        """Logical length: what remains after pending evictions. The next token's position."""
        return min(self.length, self.capacity)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        return self.capacity

    def kept(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The logical cache (pending evictions applied), without mutating the layer."""
        n, cap, s = self.length, self.capacity, self.n_sink
        k, v = self._kbuf[:, :, :n], self._vbuf[:, :, :n]
        drop = n - cap
        if drop <= 0:
            return k, v
        wk = k[:, :, s + drop:]
        wk = (self._rerotate(wk, drop) if self.rerotate_keys else wk.float()).to(self.dtype)
        return torch.cat((k[:, :, :s], wk), -2), torch.cat((v[:, :, :s], v[:, :, s + drop:]), -2)

    def reset(self) -> None:
        self.length = self.tokens_seen = self.tokens_evicted = 0
        self.keys = self.values = None
        self._kbuf = self._vbuf = None
        self.is_initialized = False

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError("SinkLayer evicts tokens permanently and cannot be cropped")

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("beam search is not supported")


def find_inv_freq(model) -> torch.Tensor:
    """The RoPE inverse frequencies the model actually uses (after any rope scaling such as llama3)."""
    for module in model.modules():
        if hasattr(module, "inv_freq") and hasattr(module, "attention_scaling"):
            return module.inv_freq.detach().clone()
    raise ValueError("no rotary embedding module with `inv_freq` found in model")


class SinkCache(Cache):
    """StreamingLLM cache: `n_sink` attention sinks + a window of the `window` most recent tokens.

        cache = SinkCache.from_model(model, n_sink=4, window=1020)
        out = model(input_ids, past_key_values=cache, use_cache=True)

    `n_sink=0` gives a plain sliding window with the same eviction and re-rotation.
    `rerotate_keys=False` keeps stale rotations: the classic position bug, kept for ablation.
    `use_triton`: fused in-place eviction kernel (default: when on CUDA with triton and full rotary).
    """

    def __init__(self, n_sink: int, window: int, inv_freq: torch.Tensor, num_layers: int,
                 rerotate_keys: bool = True, use_triton: bool | None = None):
        layers = [SinkLayer(n_sink, window, inv_freq, rerotate_keys, use_triton) for _ in range(num_layers)]
        super().__init__(layers=layers)
        self.n_sink, self.window = n_sink, window

    @classmethod
    def from_model(cls, model, n_sink: int = 4, window: int = 1020, **kwargs) -> "SinkCache":
        config = model.config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers
        if any(t != "full_attention" for t in layer_types):
            raise ValueError(f"SinkCache supports full-attention models only, got layer types {set(layer_types)}")
        return cls(n_sink, window, find_inv_freq(model), config.num_hidden_layers, **kwargs)

    @property
    def capacity(self) -> int:
        return self.n_sink + self.window

    @property
    def is_steady(self) -> bool:
        """Full, with exactly one pending eviction and a minimal buffer: every further one-token
        step is identical in shape and position (safe to capture in a CUDA graph)."""
        layer = self.layers[0]
        return (layer.is_initialized and layer.tokens_evicted > 0
                and layer.length == self.capacity + 1 == layer._kbuf.shape[-2])

    @property
    def tokens_seen(self) -> int:
        return self.layers[0].tokens_seen

    @property
    def tokens_evicted(self) -> int:
        return self.layers[0].tokens_evicted
