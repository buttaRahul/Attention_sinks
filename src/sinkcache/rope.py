"""RoPE helpers, using the same convention as Transformers (Llama/Qwen).

HF's rotary embedding works on the "half-split" layout: for a head of size d,
dimension i is paired with dimension i + d/2, and the pair is rotated by angle
pos * inv_freq[i]:

    rope(x, p) = x * cos(p*theta) + rotate_half(x) * sin(p*theta)

A rotation by p followed by a rotation by q is a rotation by p + q, because
2-D rotations compose by adding angles. So a key that was rotated at position p
and is then rotated by -delta is exactly the key rotated at p - delta. That is
what lets the sink cache "move" kept keys to new positions after an eviction
without recomputing them.
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape [..., rot_dim] for integer `positions`, computed in fp32 like HF does."""
    freqs = positions.to(torch.float32)[..., None] * inv_freq.to(torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first cos.shape[-1] dims of x (partial rotary leaves the rest untouched).
    Math is done in fp32 and rounded to x.dtype once."""
    rot_dim = cos.shape[-1]
    xf = x.to(torch.float32)
    x_rot, x_pass = xf[..., :rot_dim], xf[..., rot_dim:]
    out = x_rot * cos + rotate_half(x_rot) * sin
    if x_pass.shape[-1]:
        out = torch.cat((out, x_pass), dim=-1)
    return out.to(x.dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotate x of shape [batch, heads, seq, head_dim] at `positions` (shape [seq])."""
    cos, sin = rope_cos_sin(positions.to(x.device), inv_freq.to(x.device))
    return _rotate(x, cos, sin)


def rerotate(keys: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Shift already-rotated keys `delta` positions earlier: a rotation by -delta.

    Every key gets the same shift, so cos/sin is a single [rot_dim] vector."""
    cos, sin = rope_cos_sin(torch.tensor(-delta, device=keys.device), inv_freq.to(keys.device))
    return _rotate(keys, cos, sin)
