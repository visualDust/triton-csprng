from __future__ import annotations

import torch
import triton
import triton.language as tl

_CHACHA20_CONST = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
_MASK32 = 0xFFFFFFFF


def _as_uint32_tensor(
    values: torch.Tensor | list[int] | tuple[int, ...],
    *,
    device: torch.device | str,
    length: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.uint32, device=device)
    if tensor.numel() != length:
        raise ValueError(f"{name} must contain {length} uint32 words")
    return tensor.contiguous()


def make_chacha20_state(
    *,
    num_blocks: int,
    key: torch.Tensor | list[int] | tuple[int, ...],
    nonce: torch.Tensor | list[int] | tuple[int, ...],
    counter: int = 0,
    device: torch.device | str = "cuda:0",
) -> torch.Tensor:
    """Build `[num_blocks, 16]` ChaCha20 states.

    State layout: constants in words 0..3, 256-bit key in 4..11,
    64-bit counter in 12..13, and 64-bit nonce in 14..15.
    """

    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    key_t = _as_uint32_tensor(key, device=device, length=8, name="key")
    nonce_t = _as_uint32_tensor(nonce, device=device, length=2, name="nonce")
    state = torch.zeros((num_blocks, 16), dtype=torch.uint32, device=device)
    state[:, 0:4] = torch.tensor(
        _CHACHA20_CONST, dtype=torch.uint32, device=device
    )
    state[:, 4:12] = key_t[None, :]
    counters = torch.arange(
        counter, counter + num_blocks, dtype=torch.int64, device=device
    )
    state[:, 12] = (counters & _MASK32).to(torch.uint32)
    state[:, 13] = ((counters >> 32) & _MASK32).to(torch.uint32)
    state[:, 14:16] = nonce_t[None, :]
    return state.contiguous()


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
def _chacha20_blocks_kernel(
    states,
    out,
    n_blocks: tl.constexpr,
    step: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_blocks
    base = offs * 16

    s0 = tl.load(states + base + 0, mask=mask, other=0).to(tl.uint32)
    s1 = tl.load(states + base + 1, mask=mask, other=0).to(tl.uint32)
    s2 = tl.load(states + base + 2, mask=mask, other=0).to(tl.uint32)
    s3 = tl.load(states + base + 3, mask=mask, other=0).to(tl.uint32)
    s4 = tl.load(states + base + 4, mask=mask, other=0).to(tl.uint32)
    s5 = tl.load(states + base + 5, mask=mask, other=0).to(tl.uint32)
    s6 = tl.load(states + base + 6, mask=mask, other=0).to(tl.uint32)
    s7 = tl.load(states + base + 7, mask=mask, other=0).to(tl.uint32)
    s8 = tl.load(states + base + 8, mask=mask, other=0).to(tl.uint32)
    s9 = tl.load(states + base + 9, mask=mask, other=0).to(tl.uint32)
    s10 = tl.load(states + base + 10, mask=mask, other=0).to(tl.uint32)
    s11 = tl.load(states + base + 11, mask=mask, other=0).to(tl.uint32)
    s12 = tl.load(states + base + 12, mask=mask, other=0).to(tl.uint32)
    s13 = tl.load(states + base + 13, mask=mask, other=0).to(tl.uint32)
    s14 = tl.load(states + base + 14, mask=mask, other=0).to(tl.uint32)
    s15 = tl.load(states + base + 15, mask=mask, other=0).to(tl.uint32)

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

    o0 = (x0 + s0) & 0xFFFFFFFF
    o1 = (x1 + s1) & 0xFFFFFFFF
    o2 = (x2 + s2) & 0xFFFFFFFF
    o3 = (x3 + s3) & 0xFFFFFFFF
    o4 = (x4 + s4) & 0xFFFFFFFF
    o5 = (x5 + s5) & 0xFFFFFFFF
    o6 = (x6 + s6) & 0xFFFFFFFF
    o7 = (x7 + s7) & 0xFFFFFFFF
    o8 = (x8 + s8) & 0xFFFFFFFF
    o9 = (x9 + s9) & 0xFFFFFFFF
    o10 = (x10 + s10) & 0xFFFFFFFF
    o11 = (x11 + s11) & 0xFFFFFFFF
    o12 = (x12 + s12) & 0xFFFFFFFF
    o13 = (x13 + s13) & 0xFFFFFFFF
    o14 = (x14 + s14) & 0xFFFFFFFF
    o15 = (x15 + s15) & 0xFFFFFFFF

    tl.store(out + base + 0, o0, mask=mask)
    tl.store(out + base + 1, o1, mask=mask)
    tl.store(out + base + 2, o2, mask=mask)
    tl.store(out + base + 3, o3, mask=mask)
    tl.store(out + base + 4, o4, mask=mask)
    tl.store(out + base + 5, o5, mask=mask)
    tl.store(out + base + 6, o6, mask=mask)
    tl.store(out + base + 7, o7, mask=mask)
    tl.store(out + base + 8, o8, mask=mask)
    tl.store(out + base + 9, o9, mask=mask)
    tl.store(out + base + 10, o10, mask=mask)
    tl.store(out + base + 11, o11, mask=mask)
    tl.store(out + base + 12, o12, mask=mask)
    tl.store(out + base + 13, o13, mask=mask)
    tl.store(out + base + 14, o14, mask=mask)
    tl.store(out + base + 15, o15, mask=mask)

    if step != 0:
        old = s12
        new_low = (old + step) & 0xFFFFFFFF
        carry = new_low < old
        new_high = (s13 + carry) & 0xFFFFFFFF
        tl.store(states + base + 12, new_low, mask=mask)
        tl.store(states + base + 13, new_high, mask=mask)


def chacha20_blocks(
    states: torch.Tensor,
    *,
    step: int = 0,
    block_size: int = 128,
) -> torch.Tensor:
    """Generate ChaCha20 output blocks from `[num_blocks, 16]` states.

    `states` is updated in-place by `step` counter increments when `step != 0`.
    The returned tensor has dtype `torch.uint32` and the same shape as `states`.
    """

    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError("states must have shape [num_blocks, 16]")
    if states.dtype is not torch.uint32:
        raise TypeError("states must use torch.uint32")
    if not states.is_cuda:
        raise ValueError("Triton CSPRNG kernels require CUDA tensors")
    states = states.contiguous()
    out = torch.empty_like(states)
    n_blocks = states.shape[0]
    grid = (triton.cdiv(n_blocks, block_size),)
    with torch.cuda.device(states.device):
        _chacha20_blocks_kernel[grid](
            states, out, n_blocks, step, BLOCK=block_size
        )
    return out
