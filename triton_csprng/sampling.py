from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import mpmath as mpm
import torch
import triton
import triton.language as tl

from .stream import ChaCha20Rng


@triton.jit
def _rotl32(x, n: tl.constexpr):
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


@triton.jit
def _quarter_round(a, b, c, d):
    a = (a + b) & 0xFFFFFFFF
    d = _rotl32(d ^ a, 16)
    c = (c + d) & 0xFFFFFFFF
    b = _rotl32(b ^ c, 12)
    a = (a + b) & 0xFFFFFFFF
    d = _rotl32(d ^ a, 8)
    c = (c + d) & 0xFFFFFFFF
    b = _rotl32(b ^ c, 7)
    return a, b, c, d


@triton.jit
def _chacha20_block_words(
    block_offsets,
    state_words,
    counter_low,
    counter_high,
    BLOCK: tl.constexpr,
):
    s0 = tl.full((BLOCK,), 0x61707865, dtype=tl.uint32)
    s1 = tl.full((BLOCK,), 0x3320646E, dtype=tl.uint32)
    s2 = tl.full((BLOCK,), 0x79622D32, dtype=tl.uint32)
    s3 = tl.full((BLOCK,), 0x6B206574, dtype=tl.uint32)
    s4 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 0)).to(
        tl.uint32
    )
    s5 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 1)).to(
        tl.uint32
    )
    s6 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 2)).to(
        tl.uint32
    )
    s7 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 3)).to(
        tl.uint32
    )
    s8 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 4)).to(
        tl.uint32
    )
    s9 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 5)).to(
        tl.uint32
    )
    s10 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 6)).to(
        tl.uint32
    )
    s11 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 7)).to(
        tl.uint32
    )

    counter = (
        tl.zeros((BLOCK,), dtype=tl.uint64) + counter_low
    ) + block_offsets
    s12 = (counter & 0xFFFFFFFF).to(tl.uint32)
    s13 = (
        (tl.zeros((BLOCK,), dtype=tl.uint64) + counter_high) + (counter >> 32)
    ).to(tl.uint32)
    s14 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 8)).to(
        tl.uint32
    )
    s15 = (tl.zeros((BLOCK,), dtype=tl.uint32) + tl.load(state_words + 9)).to(
        tl.uint32
    )

    x0 = s0
    x1 = s1
    x2 = s2
    x3 = s3
    x4 = s4
    x5 = s5
    x6 = s6
    x7 = s7
    x8 = s8
    x9 = s9
    x10 = s10
    x11 = s11
    x12 = s12
    x13 = s13
    x14 = s14
    x15 = s15

    for _ in tl.static_range(10):
        x0, x4, x8, x12 = _quarter_round(x0, x4, x8, x12)
        x1, x5, x9, x13 = _quarter_round(x1, x5, x9, x13)
        x2, x6, x10, x14 = _quarter_round(x2, x6, x10, x14)
        x3, x7, x11, x15 = _quarter_round(x3, x7, x11, x15)
        x0, x5, x10, x15 = _quarter_round(x0, x5, x10, x15)
        x1, x6, x11, x12 = _quarter_round(x1, x6, x11, x12)
        x2, x7, x8, x13 = _quarter_round(x2, x7, x8, x13)
        x3, x4, x9, x14 = _quarter_round(x3, x4, x9, x14)

    return (
        (x0 + s0) & 0xFFFFFFFF,
        (x1 + s1) & 0xFFFFFFFF,
        (x2 + s2) & 0xFFFFFFFF,
        (x3 + s3) & 0xFFFFFFFF,
        (x4 + s4) & 0xFFFFFFFF,
        (x5 + s5) & 0xFFFFFFFF,
        (x6 + s6) & 0xFFFFFFFF,
        (x7 + s7) & 0xFFFFFFFF,
        (x8 + s8) & 0xFFFFFFFF,
        (x9 + s9) & 0xFFFFFFFF,
        (x10 + s10) & 0xFFFFFFFF,
        (x11 + s11) & 0xFFFFFFFF,
        (x12 + s12) & 0xFFFFFFFF,
        (x13 + s13) & 0xFFFFFFFF,
        (x14 + s14) & 0xFFFFFFFF,
        (x15 + s15) & 0xFFFFFFFF,
    )


@triton.jit
def _chacha20_block_words_two_streams(
    block_offsets,
    n_first_blocks,
    state_words_first,
    counter_low_first,
    counter_high_first,
    state_words_second,
    counter_low_second,
    counter_high_second,
    BLOCK: tl.constexpr,
):
    use_second = block_offsets >= n_first_blocks
    local_offsets = tl.where(
        use_second, block_offsets - n_first_blocks, block_offsets
    )

    s0 = tl.full((BLOCK,), 0x61707865, dtype=tl.uint32)
    s1 = tl.full((BLOCK,), 0x3320646E, dtype=tl.uint32)
    s2 = tl.full((BLOCK,), 0x79622D32, dtype=tl.uint32)
    s3 = tl.full((BLOCK,), 0x6B206574, dtype=tl.uint32)
    z32 = tl.zeros((BLOCK,), dtype=tl.uint32)
    z64 = tl.zeros((BLOCK,), dtype=tl.uint64)
    s4 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 0),
        z32 + tl.load(state_words_first + 0),
    ).to(tl.uint32)
    s5 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 1),
        z32 + tl.load(state_words_first + 1),
    ).to(tl.uint32)
    s6 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 2),
        z32 + tl.load(state_words_first + 2),
    ).to(tl.uint32)
    s7 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 3),
        z32 + tl.load(state_words_first + 3),
    ).to(tl.uint32)
    s8 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 4),
        z32 + tl.load(state_words_first + 4),
    ).to(tl.uint32)
    s9 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 5),
        z32 + tl.load(state_words_first + 5),
    ).to(tl.uint32)
    s10 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 6),
        z32 + tl.load(state_words_first + 6),
    ).to(tl.uint32)
    s11 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 7),
        z32 + tl.load(state_words_first + 7),
    ).to(tl.uint32)

    counter_low = tl.where(
        use_second, z64 + counter_low_second, z64 + counter_low_first
    )
    counter_high = tl.where(
        use_second, z64 + counter_high_second, z64 + counter_high_first
    )
    counter = counter_low + local_offsets
    s12 = (counter & 0xFFFFFFFF).to(tl.uint32)
    s13 = (counter_high + (counter >> 32)).to(tl.uint32)
    s14 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 8),
        z32 + tl.load(state_words_first + 8),
    ).to(tl.uint32)
    s15 = tl.where(
        use_second,
        z32 + tl.load(state_words_second + 9),
        z32 + tl.load(state_words_first + 9),
    ).to(tl.uint32)

    x0 = s0
    x1 = s1
    x2 = s2
    x3 = s3
    x4 = s4
    x5 = s5
    x6 = s6
    x7 = s7
    x8 = s8
    x9 = s9
    x10 = s10
    x11 = s11
    x12 = s12
    x13 = s13
    x14 = s14
    x15 = s15

    for _ in tl.static_range(10):
        x0, x4, x8, x12 = _quarter_round(x0, x4, x8, x12)
        x1, x5, x9, x13 = _quarter_round(x1, x5, x9, x13)
        x2, x6, x10, x14 = _quarter_round(x2, x6, x10, x14)
        x3, x7, x11, x15 = _quarter_round(x3, x7, x11, x15)
        x0, x5, x10, x15 = _quarter_round(x0, x5, x10, x15)
        x1, x6, x11, x12 = _quarter_round(x1, x6, x11, x12)
        x2, x7, x8, x13 = _quarter_round(x2, x7, x8, x13)
        x3, x4, x9, x14 = _quarter_round(x3, x4, x9, x14)

    return (
        (x0 + s0) & 0xFFFFFFFF,
        (x1 + s1) & 0xFFFFFFFF,
        (x2 + s2) & 0xFFFFFFFF,
        (x3 + s3) & 0xFFFFFFFF,
        (x4 + s4) & 0xFFFFFFFF,
        (x5 + s5) & 0xFFFFFFFF,
        (x6 + s6) & 0xFFFFFFFF,
        (x7 + s7) & 0xFFFFFFFF,
        (x8 + s8) & 0xFFFFFFFF,
        (x9 + s9) & 0xFFFFFFFF,
        (x10 + s10) & 0xFFFFFFFF,
        (x11 + s11) & 0xFFFFFFFF,
        (x12 + s12) & 0xFFFFFFFF,
        (x13 + s13) & 0xFFFFFFFF,
        (x14 + s14) & 0xFFFFFFFF,
        (x15 + s15) & 0xFFFFFFFF,
    )


@triton.jit
def _sample_multiply_high(q, x0, x1, x2, x3, BLOCK: tl.constexpr):
    mask32 = tl.full((BLOCK,), 0xFFFFFFFF, dtype=tl.uint64)
    q = q.to(tl.uint64)
    x0 = x0.to(tl.uint64)
    x1 = x1.to(tl.uint64)
    x2 = x2.to(tl.uint64)
    x3 = x3.to(tl.uint64)
    pl = q & mask32
    ph = q >> 32

    xl = x1
    xh = x0
    plxl = pl * xl
    plxh = pl * xh
    phxl = ph * xl
    phxh = ph * xh
    carry = (plxl >> 32) + (plxh & mask32) + (phxl & mask32)
    alpha = phxh + (plxh >> 32) + (phxl >> 32) + (carry >> 32)

    xhh = x2
    xhl = x3
    plxhl = pl * xhl
    plxhh = pl * xhh
    phxhl = ph * xhl
    phxhh = ph * xhh
    carry = ((plxhl & mask32) + (alpha & mask32)) >> 32
    carry = (
        carry
        + (plxhl >> 32)
        + (alpha >> 32)
        + (phxhl & mask32)
        + (plxhh & mask32)
    ) >> 32
    return carry + (phxhl >> 32) + (plxhh >> 32) + phxhh


@triton.jit
def _fused_bounded_uint64_kernel(
    bounds,
    out,
    n_values,
    n_blocks,
    values_per_bound,
    state_words,
    counter_low,
    counter_high,
    BLOCK: tl.constexpr,
):
    pid_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    block_offsets = pid_offsets
    block_mask = block_offsets < n_blocks
    (
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        o8,
        o9,
        o10,
        o11,
        o12,
        o13,
        o14,
        o15,
    ) = _chacha20_block_words(
        pid_offsets,
        state_words,
        counter_low,
        counter_high,
        BLOCK,
    )

    for group in tl.static_range(0, 4):
        sample_idx = block_offsets * 4 + group
        mask = block_mask & (sample_idx < n_values)
        bound_idx = sample_idx // values_per_bound
        q = tl.load(bounds + bound_idx, mask=mask, other=1).to(tl.uint64)
        if group == 0:
            sample = _sample_multiply_high(q, o0, o1, o2, o3, BLOCK)
        elif group == 1:
            sample = _sample_multiply_high(q, o4, o5, o6, o7, BLOCK)
        elif group == 2:
            sample = _sample_multiply_high(q, o8, o9, o10, o11, BLOCK)
        else:
            sample = _sample_multiply_high(q, o12, o13, o14, o15, BLOCK)
        tl.store(out + sample_idx, sample.to(tl.int64), mask=mask)


@triton.jit
def _fused_discrete_gaussian_kernel(
    thresholds_low,
    thresholds_high,
    out,
    n_values,
    n_blocks,
    table_size: tl.constexpr,
    state_words,
    counter_low,
    counter_high,
    BLOCK: tl.constexpr,
):
    pid_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    block_offsets = pid_offsets
    block_mask = block_offsets < n_blocks
    (
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        o8,
        o9,
        o10,
        o11,
        o12,
        o13,
        o14,
        o15,
    ) = _chacha20_block_words(
        pid_offsets,
        state_words,
        counter_low,
        counter_high,
        BLOCK,
    )

    for group in tl.static_range(0, 4):
        sample_idx = block_offsets * 4 + group
        mask = block_mask & (sample_idx < n_values)
        if group == 0:
            x0 = o0.to(tl.uint64)
            x1 = o1.to(tl.uint64)
            x2 = o2.to(tl.uint64)
            x3 = o3.to(tl.uint64)
        elif group == 1:
            x0 = o4.to(tl.uint64)
            x1 = o5.to(tl.uint64)
            x2 = o6.to(tl.uint64)
            x3 = o7.to(tl.uint64)
        elif group == 2:
            x0 = o8.to(tl.uint64)
            x1 = o9.to(tl.uint64)
            x2 = o10.to(tl.uint64)
            x3 = o11.to(tl.uint64)
        else:
            x0 = o12.to(tl.uint64)
            x1 = o13.to(tl.uint64)
            x2 = o14.to(tl.uint64)
            x3 = o15.to(tl.uint64)
        rnd_low = (x0 << 32) | x1
        rnd_high_with_sign = (x2 << 32) | x3
        sign_bit = rnd_high_with_sign & 1
        rnd_high = rnd_high_with_sign >> 1
        mag = tl.zeros((BLOCK,), dtype=tl.int64)
        for i in tl.static_range(0, table_size):
            threshold_low = tl.load(thresholds_low + i).to(tl.uint64)
            threshold_high = tl.load(thresholds_high + i).to(tl.uint64)
            ge = (rnd_high > threshold_high) | (
                (rnd_high == threshold_high) & (rnd_low >= threshold_low)
            )
            mag += ge.to(tl.int64)
        signed = tl.where(sign_bit == 0, -mag, mag)
        tl.store(out + sample_idx, signed, mask=mask)


@triton.jit
def _fused_stochastic_round_kernel(
    values,
    out,
    n_values,
    n_blocks,
    state_words,
    counter_low,
    counter_high,
    BLOCK: tl.constexpr,
):
    pid_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    block_offsets = pid_offsets
    block_mask = block_offsets < n_blocks
    (
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        o8,
        o9,
        o10,
        o11,
        o12,
        o13,
        o14,
        o15,
    ) = _chacha20_block_words(
        pid_offsets,
        state_words,
        counter_low,
        counter_high,
        BLOCK,
    )

    for group in tl.static_range(0, 16):
        sample_idx = block_offsets * 16 + group
        mask = block_mask & (sample_idx < n_values)
        if group == 0:
            rnd = o0
        elif group == 1:
            rnd = o1
        elif group == 2:
            rnd = o2
        elif group == 3:
            rnd = o3
        elif group == 4:
            rnd = o4
        elif group == 5:
            rnd = o5
        elif group == 6:
            rnd = o6
        elif group == 7:
            rnd = o7
        elif group == 8:
            rnd = o8
        elif group == 9:
            rnd = o9
        elif group == 10:
            rnd = o10
        elif group == 11:
            rnd = o11
        elif group == 12:
            rnd = o12
        elif group == 13:
            rnd = o13
        elif group == 14:
            rnd = o14
        else:
            rnd = o15
        x = tl.load(values + sample_idx, mask=mask, other=0.0).to(tl.float64)
        sign = x < 0.0
        ax = tl.abs(x)
        base_f = tl.floor(ax)
        frac = ax - base_f
        threshold = (frac * 4294967296.0).to(tl.uint32)
        rounded = base_f.to(tl.int64) + (rnd < threshold).to(tl.int64)
        rounded = tl.where(sign, -rounded, rounded)
        tl.store(out + sample_idx, rounded, mask=mask)


@triton.jit
def _fused_bounded_uint64_two_streams_kernel(
    bounds,
    out,
    n_values,
    n_blocks,
    n_first_blocks,
    values_per_bound,
    state_words_first,
    counter_low_first,
    counter_high_first,
    state_words_second,
    counter_low_second,
    counter_high_second,
    BLOCK: tl.constexpr,
):
    block_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_offsets < n_blocks
    (
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        o8,
        o9,
        o10,
        o11,
        o12,
        o13,
        o14,
        o15,
    ) = _chacha20_block_words_two_streams(
        block_offsets,
        n_first_blocks,
        state_words_first,
        counter_low_first,
        counter_high_first,
        state_words_second,
        counter_low_second,
        counter_high_second,
        BLOCK,
    )
    for group in tl.static_range(0, 4):
        sample_idx = block_offsets * 4 + group
        mask = block_mask & (sample_idx < n_values)
        bound_idx = sample_idx // values_per_bound
        q = tl.load(bounds + bound_idx, mask=mask, other=1).to(tl.uint64)
        if group == 0:
            sample = _sample_multiply_high(q, o0, o1, o2, o3, BLOCK)
        elif group == 1:
            sample = _sample_multiply_high(q, o4, o5, o6, o7, BLOCK)
        elif group == 2:
            sample = _sample_multiply_high(q, o8, o9, o10, o11, BLOCK)
        else:
            sample = _sample_multiply_high(q, o12, o13, o14, o15, BLOCK)
        tl.store(out + sample_idx, sample.to(tl.int64), mask=mask)


@triton.jit
def _fused_discrete_gaussian_two_streams_kernel(
    thresholds_low,
    thresholds_high,
    out,
    n_values,
    n_blocks,
    n_first_blocks,
    table_size: tl.constexpr,
    state_words_first,
    counter_low_first,
    counter_high_first,
    state_words_second,
    counter_low_second,
    counter_high_second,
    BLOCK: tl.constexpr,
):
    block_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_offsets < n_blocks
    (
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        o8,
        o9,
        o10,
        o11,
        o12,
        o13,
        o14,
        o15,
    ) = _chacha20_block_words_two_streams(
        block_offsets,
        n_first_blocks,
        state_words_first,
        counter_low_first,
        counter_high_first,
        state_words_second,
        counter_low_second,
        counter_high_second,
        BLOCK,
    )
    for group in tl.static_range(0, 4):
        sample_idx = block_offsets * 4 + group
        mask = block_mask & (sample_idx < n_values)
        if group == 0:
            x0 = o0.to(tl.uint64)
            x1 = o1.to(tl.uint64)
            x2 = o2.to(tl.uint64)
            x3 = o3.to(tl.uint64)
        elif group == 1:
            x0 = o4.to(tl.uint64)
            x1 = o5.to(tl.uint64)
            x2 = o6.to(tl.uint64)
            x3 = o7.to(tl.uint64)
        elif group == 2:
            x0 = o8.to(tl.uint64)
            x1 = o9.to(tl.uint64)
            x2 = o10.to(tl.uint64)
            x3 = o11.to(tl.uint64)
        else:
            x0 = o12.to(tl.uint64)
            x1 = o13.to(tl.uint64)
            x2 = o14.to(tl.uint64)
            x3 = o15.to(tl.uint64)
        rnd_low = (x0 << 32) | x1
        rnd_high_with_sign = (x2 << 32) | x3
        sign_bit = rnd_high_with_sign & 1
        rnd_high = rnd_high_with_sign >> 1
        mag = tl.zeros((BLOCK,), dtype=tl.int64)
        for i in tl.static_range(0, table_size):
            threshold_low = tl.load(thresholds_low + i).to(tl.uint64)
            threshold_high = tl.load(thresholds_high + i).to(tl.uint64)
            ge = (rnd_high > threshold_high) | (
                (rnd_high == threshold_high) & (rnd_low >= threshold_low)
            )
            mag += ge.to(tl.int64)
        signed = tl.where(sign_bit == 0, -mag, mag)
        tl.store(out + sample_idx, signed, mask=mask)


@triton.jit
def _bounded_uint64_kernel(
    words,
    bounds,
    out,
    n_values,
    values_per_bound,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_values
    bound_idx = offs // values_per_bound
    q = tl.load(bounds + bound_idx, mask=mask, other=1).to(tl.uint64)
    base = offs * 4
    x0 = tl.load(words + base + 0, mask=mask, other=0).to(tl.uint64)
    x1 = tl.load(words + base + 1, mask=mask, other=0).to(tl.uint64)
    x2 = tl.load(words + base + 2, mask=mask, other=0).to(tl.uint64)
    x3 = tl.load(words + base + 3, mask=mask, other=0).to(tl.uint64)

    mask32 = tl.full((BLOCK,), 0xFFFFFFFF, dtype=tl.uint64)
    pl = q & mask32
    ph = q >> 32

    # Return floor(q * U / 2**128) from a 128-bit ChaCha word U.  Counts over
    # the finite 2**128 domain differ by at most one per output bin and there is
    # no rejection fallback path.
    xl = x1
    xh = x0
    plxl = pl * xl
    plxh = pl * xh
    phxl = ph * xl
    phxh = ph * xh
    carry = (plxl >> 32) + (plxh & mask32) + (phxl & mask32)
    alpha = phxh + (plxh >> 32) + (phxl >> 32) + (carry >> 32)

    xhh = x2
    xhl = x3
    plxhl = pl * xhl
    plxhh = pl * xhh
    phxhl = ph * xhl
    phxhh = ph * xhh
    carry = ((plxhl & mask32) + (alpha & mask32)) >> 32
    carry = (
        carry
        + (plxhl >> 32)
        + (alpha >> 32)
        + (phxhl & mask32)
        + (plxhh & mask32)
    ) >> 32
    sample = carry + (phxhl >> 32) + (plxhh >> 32) + phxhh
    tl.store(out + offs, sample.to(tl.int64), mask=mask)


@triton.jit
def _discrete_gaussian_kernel(
    words,
    thresholds_low,
    thresholds_high,
    out,
    n_values,
    table_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_values
    base = offs * 4
    x0 = tl.load(words + base + 0, mask=mask, other=0).to(tl.uint64)
    x1 = tl.load(words + base + 1, mask=mask, other=0).to(tl.uint64)
    x2 = tl.load(words + base + 2, mask=mask, other=0).to(tl.uint64)
    x3 = tl.load(words + base + 3, mask=mask, other=0).to(tl.uint64)
    rnd_low = (x0 << 32) | x1
    rnd_high_with_sign = (x2 << 32) | x3
    sign_bit = rnd_high_with_sign & 1
    rnd_high = rnd_high_with_sign >> 1
    mag = tl.zeros((BLOCK,), dtype=tl.int64)
    for i in tl.static_range(0, table_size):
        threshold_low = tl.load(thresholds_low + i).to(tl.uint64)
        threshold_high = tl.load(thresholds_high + i).to(tl.uint64)
        ge = (rnd_high > threshold_high) | (
            (rnd_high == threshold_high) & (rnd_low >= threshold_low)
        )
        mag += ge.to(tl.int64)
    signed = tl.where(sign_bit == 0, -mag, mag)
    tl.store(out + offs, signed, mask=mask)


@triton.jit
def _stochastic_round_kernel(
    values,
    words,
    out,
    n_values,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_values
    x = tl.load(values + offs, mask=mask, other=0.0).to(tl.float64)
    rnd = tl.load(words + offs, mask=mask, other=0).to(tl.uint32)
    sign = x < 0.0
    ax = tl.abs(x)
    base_f = tl.floor(ax)
    frac = ax - base_f
    threshold = (frac * 4294967296.0).to(tl.uint32)
    rounded = base_f.to(tl.int64) + (rnd < threshold).to(tl.int64)
    rounded = tl.where(sign, -rounded, rounded)
    tl.store(out + offs, rounded, mask=mask)


def _as_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(dim) for dim in shape)


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _bounds_tensor(
    bounds: int | Sequence[int] | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(bounds, torch.Tensor):
        values = [int(v) for v in bounds.detach().cpu().flatten().tolist()]
    elif isinstance(bounds, int):
        values = [int(bounds)]
    else:
        values = [int(v) for v in bounds]
    if len(values) == 0:
        raise ValueError("bounds must not be empty")
    if any(v <= 0 for v in values):
        raise ValueError("bounds must be positive")
    if any(v > torch.iinfo(torch.int64).max for v in values):
        raise ValueError("bounds must fit in signed int64 output range")
    tensor = torch.tensor(values, dtype=torch.uint64, device=device)
    return tensor.contiguous()


def _can_use_fused(rng: ChaCha20Rng, *, required_words: int) -> bool:
    return required_words % 16 == 0 and rng._pending_bytes.numel() == 0


def _rng_state_words(rng: ChaCha20Rng) -> torch.Tensor:
    state = getattr(rng, "_triton_chacha_state_words", None)
    if state is None or state.device != rng.device:
        state = torch.tensor(
            [
                *[int(v) & 0xFFFFFFFF for v in rng.key_words],
                *[int(v) & 0xFFFFFFFF for v in rng.nonce_words],
            ],
            dtype=torch.uint32,
            device=rng.device,
        ).contiguous()
        setattr(rng, "_triton_chacha_state_words", state)
    return state


def _rng_counter_args(rng: ChaCha20Rng) -> tuple[int, int]:
    counter = int(rng.counter)
    return counter & 0xFFFFFFFF, (counter >> 32) & 0xFFFFFFFF


def bounded_uint64(
    rng: ChaCha20Rng,
    bounds: int | Sequence[int] | torch.Tensor,
    shape: int | Sequence[int] | None = None,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Sample integers in `[0, bound)` as an int64 CUDA tensor.

    `bounds` may be a scalar or one bound per leading channel.  When multiple
    bounds are supplied, the output is flattened as `[num_bounds, values_per_bound]`
    internally and each row uses the corresponding bound.  The sampler matches
    the multiply-high mapping `floor(bound * U / 2**128)` from a 128-bit ChaCha
    word.  The finite-domain distribution differs by at most one preimage per
    output value and has no retry or fallback branch.
    """

    bound_t = _bounds_tensor(bounds, device=rng.device)
    if shape is None:
        out_shape = (bound_t.numel(),)
    else:
        out_shape = _as_shape(shape)
    n_values = _numel(out_shape)
    if n_values == 0:
        return torch.empty(out_shape, dtype=torch.int64, device=rng.device)
    if n_values % bound_t.numel() != 0:
        raise ValueError("output size must be divisible by number of bounds")
    values_per_bound = n_values // bound_t.numel()
    out = torch.empty((n_values,), dtype=torch.int64, device=rng.device)
    required_words = n_values * 4
    if _can_use_fused(rng, required_words=required_words):
        n_blocks = required_words // 16
        grid = (triton.cdiv(n_blocks, block_size),)
        with torch.cuda.device(rng.device):
            _fused_bounded_uint64_kernel[grid](
                bound_t,
                out,
                n_values,
                n_blocks,
                values_per_bound,
                _rng_state_words(rng),
                *_rng_counter_args(rng),
                BLOCK=block_size,
            )
        rng.counter += n_blocks
    else:
        words = rng.uint32(required_words).reshape(-1)
        grid = (triton.cdiv(n_values, block_size),)
        with torch.cuda.device(rng.device):
            _bounded_uint64_kernel[grid](
                words,
                bound_t,
                out,
                n_values,
                values_per_bound,
                BLOCK=block_size,
            )
    return out.reshape(out_shape)


def bounded_uint64_two_streams(
    first_rng: ChaCha20Rng,
    second_rng: ChaCha20Rng,
    bounds: int | Sequence[int] | torch.Tensor,
    shape: int | Sequence[int],
    *,
    first_channels: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Sample channel-major bounded integers from two streams in one kernel.

    This helper is intended for layouts whose leading channels are split into
    non-repeating and repeated channel groups.  It preserves each stream's
    contiguous channel-major counter order while avoiding two separate sampling
    launches plus a concatenation copy.
    """

    if first_rng.device != second_rng.device:
        raise ValueError("streams must live on the same device")
    if first_rng._pending_bytes.numel() or second_rng._pending_bytes.numel():
        raise ValueError("two-stream fused sampling requires aligned streams")
    out_shape = _as_shape(shape)
    if len(out_shape) < 2:
        raise ValueError("shape must be channel-major with at least 2 dims")
    n_values = _numel(out_shape)
    values_per_bound = n_values // out_shape[0]
    first_values = int(first_channels) * values_per_bound
    if n_values % out_shape[0] != 0 or first_values % 4 != 0 or n_values % 4:
        raise ValueError("two-stream fused sampling requires 4-sample blocks")
    bound_t = _bounds_tensor(bounds, device=first_rng.device)
    if bound_t.numel() != out_shape[0]:
        raise ValueError("bounds must contain one value per output channel")
    out = torch.empty((n_values,), dtype=torch.int64, device=first_rng.device)
    n_blocks = n_values // 4
    n_first_blocks = first_values // 4
    grid = (triton.cdiv(n_blocks, block_size),)
    with torch.cuda.device(first_rng.device):
        _fused_bounded_uint64_two_streams_kernel[grid](
            bound_t,
            out,
            n_values,
            n_blocks,
            n_first_blocks,
            values_per_bound,
            _rng_state_words(first_rng),
            *_rng_counter_args(first_rng),
            _rng_state_words(second_rng),
            *_rng_counter_args(second_rng),
            BLOCK=block_size,
        )
    first_rng.counter += n_first_blocks
    second_rng.counter += n_blocks - n_first_blocks
    return out.reshape(out_shape)


@functools.lru_cache(maxsize=32)
def _half_plane_cdt_threshold_words(
    sigma: float,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return 128-bit half-plane CDT thresholds.

    The sampler builds a 128-bit half-plane CDT with precision
    `2 * security_bits`, reserves one random bit for sign, and uses thresholds
    `CDT[1]..CDT[num_sampling_points-1]`.  Values beyond the last threshold fall
    into the last tail bucket.
    """

    security_bits = 128
    mpm.mp.prec = security_bits * 2
    sampling_power = math.ceil(math.log2(6 * float(sigma)))
    num_sampling_points = 2**sampling_power
    mp_sigma = mpm.mpf(str(sigma))
    mp_two = mpm.mpf("2")
    norm = mp_sigma * mpm.sqrt(mp_two * mpm.pi)
    probs = [
        mpm.exp(-(mpm.mpf(str(k)) ** 2) / (mp_two * mp_sigma**2)) / norm
        for k in range(num_sampling_points)
    ]
    probs[0] /= 2
    cdf = 0.0
    scale = mp_two ** mpm.mpf(str(security_bits))
    lows: list[int] = []
    highs: list[int] = []
    mask64 = (1 << 64) - 1
    # Runtime tree nodes use CDT[1]..CDT[num_sampling_points-1]; CDT[-1] is not
    # a rejection boundary, so the truncated tail is folded into the last value.
    for prob in probs[:-1]:
        cdf += prob
        threshold = int(cdf * scale)
        lows.append(threshold & mask64)
        highs.append((threshold >> 64) & mask64)
    return tuple(lows), tuple(highs), sampling_power


def discrete_gaussian(
    rng: ChaCha20Rng,
    shape: int | Sequence[int],
    *,
    sigma: float = 3.2,
    block_size: int = 128,
) -> torch.Tensor:
    """Sample a centered integer discrete Gaussian using half-plane CDT."""

    out_shape = _as_shape(shape)
    n_values = _numel(out_shape)
    if n_values == 0:
        return torch.empty(out_shape, dtype=torch.int64, device=rng.device)
    threshold_lows, threshold_highs, _tree_depth = (
        _half_plane_cdt_threshold_words(float(sigma))
    )
    threshold_low_t = torch.tensor(
        threshold_lows,
        dtype=torch.uint64,
        device=rng.device,
    )
    threshold_high_t = torch.tensor(
        threshold_highs,
        dtype=torch.uint64,
        device=rng.device,
    )
    out = torch.empty((n_values,), dtype=torch.int64, device=rng.device)
    required_words = n_values * 4
    if _can_use_fused(rng, required_words=required_words):
        n_blocks = required_words // 16
        grid = (triton.cdiv(n_blocks, block_size),)
        with torch.cuda.device(rng.device):
            _fused_discrete_gaussian_kernel[grid](
                threshold_low_t,
                threshold_high_t,
                out,
                n_values,
                n_blocks,
                threshold_low_t.numel(),
                _rng_state_words(rng),
                *_rng_counter_args(rng),
                BLOCK=block_size,
            )
        rng.counter += n_blocks
    else:
        words = rng.uint32(required_words).reshape(-1)
        grid = (triton.cdiv(n_values, block_size),)
        with torch.cuda.device(rng.device):
            _discrete_gaussian_kernel[grid](
                words,
                threshold_low_t,
                threshold_high_t,
                out,
                n_values,
                threshold_low_t.numel(),
                BLOCK=block_size,
            )
    return out.reshape(out_shape)


def discrete_gaussian_two_streams(
    first_rng: ChaCha20Rng,
    second_rng: ChaCha20Rng,
    shape: int | Sequence[int],
    *,
    first_channels: int,
    sigma: float = 3.2,
    block_size: int = 128,
) -> torch.Tensor:
    """Sample channel-major discrete Gaussians from two streams in one kernel."""

    if first_rng.device != second_rng.device:
        raise ValueError("streams must live on the same device")
    if first_rng._pending_bytes.numel() or second_rng._pending_bytes.numel():
        raise ValueError("two-stream fused sampling requires aligned streams")
    out_shape = _as_shape(shape)
    if len(out_shape) < 2:
        raise ValueError("shape must be channel-major with at least 2 dims")
    n_values = _numel(out_shape)
    values_per_channel = n_values // out_shape[0]
    first_values = int(first_channels) * values_per_channel
    if n_values % out_shape[0] != 0 or first_values % 4 != 0 or n_values % 4:
        raise ValueError("two-stream fused sampling requires 4-sample blocks")
    threshold_lows, threshold_highs, _tree_depth = (
        _half_plane_cdt_threshold_words(float(sigma))
    )
    threshold_low_t = torch.tensor(
        threshold_lows,
        dtype=torch.uint64,
        device=first_rng.device,
    )
    threshold_high_t = torch.tensor(
        threshold_highs,
        dtype=torch.uint64,
        device=first_rng.device,
    )
    out = torch.empty((n_values,), dtype=torch.int64, device=first_rng.device)
    n_blocks = n_values // 4
    n_first_blocks = first_values // 4
    grid = (triton.cdiv(n_blocks, block_size),)
    with torch.cuda.device(first_rng.device):
        _fused_discrete_gaussian_two_streams_kernel[grid](
            threshold_low_t,
            threshold_high_t,
            out,
            n_values,
            n_blocks,
            n_first_blocks,
            threshold_low_t.numel(),
            _rng_state_words(first_rng),
            *_rng_counter_args(first_rng),
            _rng_state_words(second_rng),
            *_rng_counter_args(second_rng),
            BLOCK=block_size,
        )
    first_rng.counter += n_first_blocks
    second_rng.counter += n_blocks - n_first_blocks
    return out.reshape(out_shape)


def stochastic_round(
    rng: ChaCha20Rng,
    values: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Randomly round floating values to int64."""

    if not values.is_cuda:
        raise ValueError("values must be a CUDA tensor")
    if values.device != rng.device:
        raise ValueError("values must be on the same device as rng")
    values = values.contiguous()
    n_values = values.numel()
    if n_values == 0:
        return torch.empty_like(values, dtype=torch.int64)
    out = torch.empty((n_values,), dtype=torch.int64, device=rng.device)
    if _can_use_fused(rng, required_words=n_values):
        n_blocks = n_values // 16
        grid = (triton.cdiv(n_blocks, block_size),)
        with torch.cuda.device(rng.device):
            _fused_stochastic_round_kernel[grid](
                values.reshape(-1),
                out,
                n_values,
                n_blocks,
                _rng_state_words(rng),
                *_rng_counter_args(rng),
                BLOCK=block_size,
            )
        rng.counter += n_blocks
    else:
        words = rng.uint32(n_values).reshape(-1)
        grid = (triton.cdiv(n_values, block_size),)
        with torch.cuda.device(rng.device):
            _stochastic_round_kernel[grid](
                values.reshape(-1), words, out, n_values, BLOCK=block_size
            )
    return out.reshape(values.shape)
