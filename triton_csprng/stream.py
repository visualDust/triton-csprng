from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch

from .chacha20 import chacha20_blocks, make_chacha20_state

_UINT32_WORD_BYTES = 4
_CHACHA20_BLOCK_WORDS = 16
_CHACHA20_BLOCK_BYTES = _CHACHA20_BLOCK_WORDS * _UINT32_WORD_BYTES


def _random_uint32_words(count: int) -> list[int]:
    data = os.urandom(count * _UINT32_WORD_BYTES)
    return [
        int.from_bytes(data[i : i + _UINT32_WORD_BYTES], "little")
        for i in range(0, len(data), _UINT32_WORD_BYTES)
    ]


def _normalize_words(
    value: Sequence[int] | bytes | bytearray | torch.Tensor | None,
    *,
    n_words: int,
    name: str,
) -> list[int]:
    if value is None:
        return _random_uint32_words(n_words)
    if isinstance(value, torch.Tensor):
        words = [int(v) for v in value.detach().cpu().flatten().tolist()]
    elif isinstance(value, bytes | bytearray):
        if len(value) != n_words * _UINT32_WORD_BYTES:
            raise ValueError(
                f"{name} must be {n_words * _UINT32_WORD_BYTES} bytes"
            )
        words = [
            int.from_bytes(value[i : i + _UINT32_WORD_BYTES], "little")
            for i in range(0, len(value), _UINT32_WORD_BYTES)
        ]
    else:
        words = [int(v) for v in value]
    if len(words) != n_words:
        raise ValueError(f"{name} must contain {n_words} uint32 words")
    return [v & 0xFFFFFFFF for v in words]


def _numel(shape: int | Sequence[int]) -> int:
    if isinstance(shape, int):
        return shape
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


class ChaCha20Rng:
    """Counter-based ChaCha20 stream backed by Triton kernels.

    This is intentionally a small low-level primitive.  Callers own their
    stream layout and request raw uint32 words or bytes from an explicit
    key/nonce/counter tuple.
    """

    def __init__(
        self,
        *,
        key: Sequence[int] | bytes | bytearray | torch.Tensor | None = None,
        nonce: Sequence[int] | bytes | bytearray | torch.Tensor | None = None,
        counter: int = 0,
        device: torch.device | str = "cuda:0",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("ChaCha20Rng currently requires a CUDA device")
        self.key_words = _normalize_words(key, n_words=8, name="key")
        self.nonce_words = _normalize_words(nonce, n_words=2, name="nonce")
        self.counter = int(counter)
        self._pending_bytes = torch.empty(
            (0,), dtype=torch.uint8, device=self.device
        )

    def blocks(self, num_blocks: int) -> torch.Tensor:
        """Return `[num_blocks, 16]` uint32 ChaCha20 output blocks."""

        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        state = make_chacha20_state(
            num_blocks=num_blocks,
            key=self.key_words,
            nonce=self.nonce_words,
            counter=self.counter,
            device=self.device,
        )
        out = chacha20_blocks(state, step=0)
        self.counter += num_blocks
        return out

    def uint32(self, shape: int | Sequence[int]) -> torch.Tensor:
        """Return random uint32 words with `shape`."""

        n_words = _numel(shape)
        if n_words < 0:
            raise ValueError("shape must have non-negative size")
        if n_words == 0:
            return torch.empty(shape, dtype=torch.uint32, device=self.device)
        words = self._next_bytes(n_words * _UINT32_WORD_BYTES).view(
            torch.uint32
        )
        return words.reshape(shape)

    def bytes(self, shape: int | Sequence[int]) -> torch.Tensor:
        """Return random bytes with `shape` as a `torch.uint8` CUDA tensor."""

        n_bytes = _numel(shape)
        if n_bytes < 0:
            raise ValueError("shape must have non-negative size")
        if n_bytes == 0:
            return torch.empty(shape, dtype=torch.uint8, device=self.device)
        return self._next_bytes(n_bytes).reshape(shape)

    def randint(
        self,
        bounds: int | Sequence[int] | torch.Tensor,
        shape: int | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Sample bounded integers in `[0, bound)`.

        This convenience method delegates to `triton_csprng.sampling` to avoid
        making the stream class own every distribution-specific kernel.
        """

        from .sampling import bounded_uint64

        return bounded_uint64(self, bounds, shape)

    def discrete_gaussian(
        self,
        shape: int | Sequence[int],
        *,
        sigma: float = 3.2,
    ) -> torch.Tensor:
        """Sample a centered integer discrete Gaussian."""

        from .sampling import discrete_gaussian

        return discrete_gaussian(self, shape, sigma=sigma)

    def stochastic_round(self, values: torch.Tensor) -> torch.Tensor:
        """Randomly round floating values to int64."""

        from .sampling import stochastic_round

        return stochastic_round(self, values)

    def _next_bytes(self, n_bytes: int) -> torch.Tensor:
        chunks = []
        remaining = n_bytes
        if self._pending_bytes.numel() > 0:
            take = min(remaining, self._pending_bytes.numel())
            chunks.append(self._pending_bytes[:take])
            self._pending_bytes = self._pending_bytes[take:].contiguous()
            remaining -= take
        if remaining > 0:
            n_blocks = (
                remaining + _CHACHA20_BLOCK_BYTES - 1
            ) // _CHACHA20_BLOCK_BYTES
            block_bytes = self.blocks(n_blocks).reshape(-1).view(torch.uint8)
            chunks.append(block_bytes[:remaining])
            self._pending_bytes = block_bytes[remaining:].contiguous()
        if len(chunks) == 1:
            return chunks[0].contiguous()
        return torch.cat(chunks, dim=0).contiguous()

    def state_dict(self) -> dict[str, Any]:
        return {
            "key_words": list(self.key_words),
            "nonce_words": list(self.nonce_words),
            "counter": self.counter,
            "device": str(self.device),
            "pending_bytes": self._pending_bytes.detach().clone(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ChaCha20Rng":
        rng = cls(
            key=state["key_words"],
            nonce=state["nonce_words"],
            counter=state["counter"],
            device=state["device"],
        )
        pending = state.get("pending_bytes")
        if pending is not None:
            rng._pending_bytes = pending.to(
                rng.device, dtype=torch.uint8
            ).contiguous()
        return rng
