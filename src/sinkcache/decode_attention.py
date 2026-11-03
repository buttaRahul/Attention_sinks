"""Split-K flash-decoding for one query token, with GQA, in Triton.

Why this exists: decode attention should read every cached K and V exactly once
(that is what makes decode memory-bound and its cost linear in cache length),
but on the Windows PyTorch build used here no SDPA backend does that for a
growing GQA cache:

  - flash attention: not compiled into Windows wheels
  - memory-efficient: no GQA, so HF repeats K/V (an extra copy) and it reaches ~37 GB/s
  - cuDNN: fast for a fixed shape, but builds a new plan for every new length (~0.5 s/token)
  - math (what HF's default dispatch lands on): upcasts to fp32, ~15x slower at 32K

This kernel splits the sequence across programs (one per KV head x split),
each program handles all query heads of its GQA group (so K/V are loaded once,
not once per query head), keeps an online softmax in fp32, and a second tiny
kernel merges the splits. It is registered as a Transformers attention backend
("sdpa_decode") that uses this kernel for single-token decode and falls back
to SDPA for prefill / multi-token chunks. Same kernel for every cache type, so
cache comparisons stay fair.
"""

import torch
import triton
import triton.language as tl

BLOCK_L = 64


@triton.jit
def _decode_partial_kernel(
    Q, K, V, O_part, M_part, L_part,
    sq_b, sq_h, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t,
    seq_len, tiles_per_split, scale,
    HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
    NS: tl.constexpr, BL: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    split = tl.program_id(1)
    b = pid_bh // HKV
    h = pid_bh % HKV
    g = tl.arange(0, GP)
    d = tl.arange(0, D)
    gm = g < G

    q = tl.load(Q + b * sq_b + (h * G + g)[:, None] * sq_h + d[None, :], mask=gm[:, None], other=0.0)
    k_base = K + b * sk_b + h * sk_h
    v_base = V + b * sv_b + h * sv_h

    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    start = split * tiles_per_split * BL
    for t in range(tiles_per_split):
        offs = start + t * BL + tl.arange(0, BL)
        km = offs < seq_len
        k = tl.load(k_base + offs[:, None] * sk_t + d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(km[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs[:, None] * sv_t + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    part = (pid_bh * NS + split) * GP + g
    tl.store(M_part + part, m_i)
    tl.store(L_part + part, l_i)
    tl.store(O_part + part[:, None] * D + d[None, :], acc)


@triton.jit
def _decode_reduce_kernel(
    O_part, M_part, L_part, Out, so_b, so_h,
    HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
    NS: tl.constexpr, NSP: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // HKV
    h = pid_bh % HKV
    sp = tl.arange(0, NSP)
    g = tl.arange(0, GP)
    d = tl.arange(0, D)
    sm = sp < NS
    idx = (pid_bh * NS + sp[:, None]) * GP + g[None, :]                     # [NSP, GP]
    m_s = tl.load(M_part + idx, mask=sm[:, None], other=float("-inf"))
    l_s = tl.load(L_part + idx, mask=sm[:, None], other=0.0)
    m = tl.max(m_s, 0)                                                     # [GP]
    w = tl.exp(m_s - m[None, :])                                           # [NSP, GP]
    l = tl.sum(w * l_s, 0)
    o = tl.zeros([GP, D], tl.float32)
    for s in range(NS):
        ws = tl.sum(tl.where(sp[:, None] == s, w, 0.0), 0)                  # [GP]
        o_s = tl.load(O_part + ((pid_bh * NS + s) * GP + g)[:, None] * D + d[None, :])
        o += ws[:, None] * o_s
    o = o / l[:, None]
    out = Out + b * so_b + (h * G + g)[:, None] * so_h + d[None, :]
    tl.store(out, o.to(Out.dtype.element_ty), mask=(g < G)[:, None])


def flash_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                 target_programs: int = 96, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    """q [B, Hq, 1, D]; k, v [B, Hkv, L, D] (any strides with unit last-dim stride).
    Returns attention output as [B, 1, Hq, D] (the layout HF attention functions return)."""
    bsz, hq, q_len, dim = q.shape
    _, hkv, seq_len, _ = k.shape
    assert q_len == 1 and hq % hkv == 0
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    group = hq // hkv
    gp = max(16, triton.next_power_of_2(group))           # tl.dot needs M >= 16
    n_tiles = triton.cdiv(seq_len, BLOCK_L)
    want = max(1, target_programs // (bsz * hkv))
    tiles_per_split = triton.cdiv(n_tiles, want)
    ns = triton.cdiv(n_tiles, tiles_per_split)            # every split gets >= 1 non-empty tile
    o_part = torch.empty(bsz * hkv * ns * gp, dim, device=q.device, dtype=torch.float32)
    m_part = torch.empty(bsz * hkv * ns * gp, device=q.device, dtype=torch.float32)
    l_part = torch.empty_like(m_part)
    out = torch.empty(bsz, 1, hq, dim, device=q.device, dtype=q.dtype)
    _decode_partial_kernel[(bsz * hkv, ns)](
        q, k, v, o_part, m_part, l_part,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        seq_len, tiles_per_split, scale,
        HKV=hkv, G=group, GP=gp, D=dim, NS=ns, BL=BLOCK_L, num_warps=num_warps, num_stages=num_stages,
    )
    _decode_reduce_kernel[(bsz * hkv,)](
        o_part, m_part, l_part, out, out.stride(0), out.stride(2),
        HKV=hkv, G=group, GP=gp, D=dim, NS=ns, NSP=triton.next_power_of_2(ns),
    )
    return out


calls = {"decode_kernel": 0, "sdpa_fallback": 0}


def decode_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Transformers attention function: Triton flash-decode for one query token, SDPA otherwise."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if query.shape[2] == 1 and attention_mask is None and query.is_cuda and not module.training:
        calls["decode_kernel"] += 1
        scale = scaling if scaling is not None else query.shape[-1] ** -0.5
        return flash_decode(query, key, value, scale), None
    calls["sdpa_fallback"] += 1
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                  scaling=scaling, **kwargs)


def register(name: str = "sdpa_decode") -> str:
    """Register the backend; then load with attn_implementation=name."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(name, decode_attention_forward)
    AttentionMaskInterface.register(name, sdpa_mask)
    return name
