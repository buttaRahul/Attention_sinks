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
position. Sinks never move. Values carry no position and are just sliced.

Order of operations in `update`: append, return everything for attention,
*then* evict. The current queries were rotated at positions
cache_len..cache_len+q_len-1, and the keys returned to attention sit at
0..cache_len+q_len-1, so query and key positions agree. Eviction only affects
what the *next* step sees. Persistent size is therefore n_sink + window, and a
decode step attends over n_sink + window + 1 keys (the same accounting for
every configuration compared here).
"""

import torch
from transformers.cache_utils import Cache, DynamicLayer

from .rope import rerotate


class SinkLayer(DynamicLayer):
    """One layer's cache: `n_sink` sink tokens + a rolling window of `window` tokens."""

    is_sliding = False      # masks treat it as a full-attention layer over what it returns
    is_croppable = False    # evicted tokens are gone; there is nothing to roll back to

    def __init__(self, n_sink: int, window: int, inv_freq: torch.Tensor, rerotate_keys: bool = True):
        super().__init__()
        if n_sink < 0 or window < 1:
            raise ValueError(f"need n_sink >= 0 and window >= 1, got {n_sink}, {window}")
        self.n_sink = n_sink
        self.window = window
        self.capacity = n_sink + window
        self.inv_freq = inv_freq
        self.rerotate_keys = rerotate_keys
        self.tokens_seen = 0
        self.tokens_evicted = 0

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        self.inv_freq = self.inv_freq.to(self.device)

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        keys = torch.cat([self.keys, key_states], dim=-2)
        values = torch.cat([self.values, value_states], dim=-2)
        self.tokens_seen += key_states.shape[-2]

        drop = keys.shape[-2] - self.capacity
        if drop > 0:
            s = self.n_sink
            window_keys = keys[:, :, s + drop:]
            if self.rerotate_keys:
                window_keys = rerotate(window_keys, drop, self.inv_freq)
            self.keys = torch.cat([keys[:, :, :s], window_keys], dim=-2)
            self.values = torch.cat([values[:, :, :s], values[:, :, s + drop:]], dim=-2)
            self.tokens_evicted += drop
        else:
            self.keys, self.values = keys, values

        # Attention for *this* step sees everything, at positions that match the queries.
        return keys, values

    def get_max_length(self) -> int:
        return self.capacity

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError("SinkLayer evicts tokens permanently and cannot be cropped")

    def reset(self) -> None:
        super().reset()
        self.tokens_seen = 0
        self.tokens_evicted = 0


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
    """

    def __init__(self, n_sink: int, window: int, inv_freq: torch.Tensor, num_layers: int,
                 rerotate_keys: bool = True):
        layers = [SinkLayer(n_sink, window, inv_freq, rerotate_keys) for _ in range(num_layers)]
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
    def tokens_seen(self) -> int:
        return self.layers[0].tokens_seen

    @property
    def tokens_evicted(self) -> int:
        return self.layers[0].tokens_evicted
