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
def _bounded_uint64_kernel(
    words,
    bounds,
    out,
    n_values: tl.constexpr,
    values_per_bound: tl.constexpr,
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
    n_values: tl.constexpr,
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
    n_values: tl.constexpr,
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


def bounded_uint64(
    rng: ChaCha20Rng,
    bounds: int | Sequence[int] | torch.Tensor,
    shape: int | Sequence[int] | None = None,
    *,
    block_size: int = 256,
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
    words = rng.uint32(n_values * 4).reshape(-1)
    out = torch.empty((n_values,), dtype=torch.int64, device=rng.device)
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
    block_size: int = 256,
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
    words = rng.uint32(n_values * 4).reshape(-1)
    out = torch.empty((n_values,), dtype=torch.int64, device=rng.device)
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


def stochastic_round(
    rng: ChaCha20Rng,
    values: torch.Tensor,
    *,
    block_size: int = 256,
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
    words = rng.uint32(n_values).reshape(-1)
    out = torch.empty((n_values,), dtype=torch.int64, device=rng.device)
    grid = (triton.cdiv(n_values, block_size),)
    with torch.cuda.device(rng.device):
        _stochastic_round_kernel[grid](
            values.reshape(-1), words, out, n_values, BLOCK=block_size
        )
    return out.reshape(values.shape)
