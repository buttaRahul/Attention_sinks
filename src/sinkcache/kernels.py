"""Fused in-place eviction for SinkLayer: shift the window `drop` slots left and
re-rotate keys by -drop, in one pass over the window.

The PyTorch version (SinkLayer._evict_torch) does it in ~6 elementwise ops with
fp32 temporaries, about 45 MB of DRAM traffic per layer per step for a
1,020-token window, which on Qwen3-0.6B is roughly as many bytes as reading
the model weights. This kernel reads each window element once and writes it
once (4 MB per layer for K+V), does the rotation in fp32 registers, and rounds
once, like the PyTorch path.

In place: destination rows [s, s+W) overlap source rows [s+drop, s+drop+W).
Each program walks its slice of rows from low to high in chunks; within a
chunk every load happens before any store (barrier), and later chunks only
read rows above anything already written, so no row is overwritten before it
has been read. Programs split the head dimension, never the rows, so they do
not touch each other's elements.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _shift_kernel(
    buf, cos, sin,
    stride_bh, stride_t,
    s, drop, W,
    HALF: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr, ROTATE: tl.constexpr,
):
    bh = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    base = buf + bh * stride_bh
    cols = d0 + tl.arange(0, BLOCK_D)          # first-half dims; partner is cols + HALF
    if ROTATE:
        c = tl.load(cos + cols)[None, :]
        sn = tl.load(sin + cols)[None, :]
    for t0 in range(0, W, BLOCK_T):
        rows = t0 + tl.arange(0, BLOCK_T)
        m = (rows < W)[:, None]
        src = base + (s + drop + rows)[:, None] * stride_t
        dst = base + (s + rows)[:, None] * stride_t
        x1 = tl.load(src + cols[None, :], mask=m)
        x2 = tl.load(src + HALF + cols[None, :], mask=m)
        tl.debug_barrier()                      # every read of this chunk before any write
        if ROTATE:
            a = x1.to(tl.float32)
            b = x2.to(tl.float32)
            # rope(x, -drop): out = x*cos + rotate_half(x)*sin, rotate_half(x) = (-x2, x1)
            y1 = a * c - b * sn
            y2 = b * c + a * sn
            tl.store(dst + cols[None, :], y1.to(x1.dtype), mask=m)
            tl.store(dst + HALF + cols[None, :], y2.to(x2.dtype), mask=m)
        else:
            tl.store(dst + cols[None, :], x1, mask=m)
            tl.store(dst + HALF + cols[None, :], x2, mask=m)


def shift_window_(buf: torch.Tensor, s: int, drop: int, window: int,
                  cos: torch.Tensor | None = None, sin: torch.Tensor | None = None) -> None:
    """In place: buf[:, :, s:s+window] = rotate(buf[:, :, s+drop:s+drop+window], -drop).
    buf is [B, H, T, D] contiguous. cos/sin are the fp32 [D] vectors for angle -drop (HF layout:
    the two halves are equal), or None to shift without rotating (values)."""
    assert buf.is_contiguous() and buf.shape[-1] % 2 == 0
    b, h, t, d = buf.shape
    half = d // 2
    block_d = min(16, half)
    rotate = cos is not None
    if not rotate:
        cos = sin = buf  # unused
    grid = (b * h, half // block_d)
    _shift_kernel[grid](buf, cos, sin, buf.stride(1), buf.stride(2), s, drop, window,
                        HALF=half, BLOCK_D=block_d, BLOCK_T=64, ROTATE=rotate)
