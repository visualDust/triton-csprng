from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .sampling import bounded_uint64, discrete_gaussian
from .stream import ChaCha20Rng


def _derive_nonce(base: Sequence[int], stream_id: int) -> list[int]:
    if len(base) != 2:
        raise ValueError("nonce must contain two uint32 words")
    return [int(base[0]) & 0xFFFFFFFF, (int(base[1]) + stream_id) & 0xFFFFFFFF]


class RnsRandomStreams:
    """Convenience stream manager for FHE-style RNS channel layouts.

    The class only knows about devices, per-device non-repeating channel counts,
    a common repeated-channel stream, and a coefficient count.  Higher-level
    packages can map their own RNS prime ordering and key/encryption semantics
    onto this substrate.
    """

    def __init__(
        self,
        *,
        num_coeffs: int,
        channel_counts: Sequence[int],
        repeated_channels: int = 0,
        devices: Sequence[str | torch.device] | None = None,
        key: Sequence[int] | bytes | bytearray | torch.Tensor | None = None,
        nonce: Sequence[int] | bytes | bytearray | torch.Tensor | None = None,
        counter: int = 0,
    ) -> None:
        if num_coeffs <= 0:
            raise ValueError("num_coeffs must be positive")
        if repeated_channels < 0:
            raise ValueError("repeated_channels must be non-negative")
        self.num_coeffs = int(num_coeffs)
        self.channel_counts = [int(v) for v in channel_counts]
        if any(v < 0 for v in self.channel_counts):
            raise ValueError("channel_counts must be non-negative")
        self.repeated_channels = int(repeated_channels)
        if devices is None:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.devices = [torch.device(device) for device in devices]
        if len(self.devices) != len(self.channel_counts):
            raise ValueError(
                "devices and channel_counts must have the same length"
            )
        if any(device.type != "cuda" for device in self.devices):
            raise ValueError("RnsRandomStreams currently requires CUDA devices")

        seed_stream = ChaCha20Rng(
            key=key, nonce=nonce, counter=counter, device=self.devices[0]
        )
        self.key_words = list(seed_stream.key_words)
        self.nonce_words = list(seed_stream.nonce_words)
        self.counter = int(counter)
        self._device_streams = [
            ChaCha20Rng(
                key=self.key_words,
                nonce=_derive_nonce(self.nonce_words, 1 + idx),
                counter=self.counter,
                device=device,
            )
            for idx, device in enumerate(self.devices)
        ]
        self._repeat_streams = [
            ChaCha20Rng(
                key=self.key_words,
                nonce=_derive_nonce(self.nonce_words, 0),
                counter=self.counter,
                device=device,
            )
            for device in self.devices
        ]

    def uint32_channels(
        self,
        channel_counts: Sequence[int] | None = None,
        *,
        repeated_channels: int | None = None,
    ) -> list[torch.Tensor]:
        """Return per-device `[channels + repeated, num_coeffs]` uint32 words."""

        counts = self._normalize_counts(channel_counts)
        repeats = (
            self.repeated_channels
            if repeated_channels is None
            else int(repeated_channels)
        )
        outputs = []
        for idx, count in enumerate(counts):
            parts = []
            if count > 0:
                parts.append(
                    self._device_streams[idx].uint32((count, self.num_coeffs))
                )
            if repeats > 0:
                parts.append(
                    self._repeat_streams[idx].uint32((repeats, self.num_coeffs))
                )
            if parts:
                outputs.append(torch.cat(parts, dim=0))
            else:
                outputs.append(
                    torch.empty(
                        (0, self.num_coeffs),
                        dtype=torch.uint32,
                        device=self.devices[idx],
                    )
                )
        return outputs

    def randint_channels(
        self,
        bounds_by_device: Sequence[Sequence[int] | torch.Tensor],
        *,
        repeated_channels: int | None = None,
    ) -> list[torch.Tensor]:
        """Return per-device bounded integer tensors.

        Each `bounds_by_device[i]` supplies one bound per output channel on
        device `i`, including any repeated channels the caller wants.
        """

        if len(bounds_by_device) != len(self.devices):
            raise ValueError("bounds_by_device length must match devices")
        repeats = (
            self.repeated_channels
            if repeated_channels is None
            else int(repeated_channels)
        )
        outputs = []
        for idx, bounds in enumerate(bounds_by_device):
            stream = self._device_streams[idx]
            bounds_t = torch.as_tensor(
                bounds, dtype=torch.uint64, device=stream.device
            ).flatten()
            if repeats > bounds_t.numel():
                raise ValueError("repeated_channels cannot exceed bound count")
            parts = []
            non_repeat_count = bounds_t.numel() - repeats
            if non_repeat_count > 0:
                non_repeat_bounds = bounds_t[:non_repeat_count]
                parts.append(
                    bounded_uint64(
                        stream,
                        non_repeat_bounds,
                        (non_repeat_bounds.numel(), self.num_coeffs),
                    )
                )
            if repeats > 0:
                repeat_bounds = bounds_t[non_repeat_count:]
                parts.append(
                    bounded_uint64(
                        self._repeat_streams[idx],
                        repeat_bounds,
                        (repeat_bounds.numel(), self.num_coeffs),
                    )
                )
            if parts:
                outputs.append(torch.cat(parts, dim=0))
            else:
                outputs.append(
                    torch.empty(
                        (0, self.num_coeffs),
                        dtype=torch.int64,
                        device=self.devices[idx],
                    )
                )
        return outputs

    def discrete_gaussian_channels(
        self,
        channel_counts: Sequence[int] | None = None,
        *,
        repeated_channels: int | None = None,
        sigma: float = 3.2,
    ) -> list[torch.Tensor]:
        """Return per-device centered discrete Gaussian channel tensors."""

        counts = self._normalize_counts(channel_counts)
        repeats = (
            self.repeated_channels
            if repeated_channels is None
            else int(repeated_channels)
        )
        outputs = []
        for idx, count in enumerate(counts):
            parts = []
            if count > 0:
                parts.append(
                    discrete_gaussian(
                        self._device_streams[idx],
                        (count, self.num_coeffs),
                        sigma=sigma,
                    )
                )
            if repeats > 0:
                parts.append(
                    discrete_gaussian(
                        self._repeat_streams[idx],
                        (repeats, self.num_coeffs),
                        sigma=sigma,
                    )
                )
            if parts:
                outputs.append(torch.cat(parts, dim=0))
            else:
                outputs.append(
                    torch.empty(
                        (0, self.num_coeffs),
                        dtype=torch.int64,
                        device=self.devices[idx],
                    )
                )
        return outputs

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_coeffs": self.num_coeffs,
            "channel_counts": list(self.channel_counts),
            "repeated_channels": self.repeated_channels,
            "devices": [str(device) for device in self.devices],
            "key_words": list(self.key_words),
            "nonce_words": list(self.nonce_words),
            "device_streams": [
                stream.state_dict() for stream in self._device_streams
            ],
            "repeat_streams": [
                stream.state_dict() for stream in self._repeat_streams
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "RnsRandomStreams":
        obj = cls(
            num_coeffs=state["num_coeffs"],
            channel_counts=state["channel_counts"],
            repeated_channels=state["repeated_channels"],
            devices=state["devices"],
            key=state["key_words"],
            nonce=state["nonce_words"],
        )
        obj._device_streams = [
            ChaCha20Rng.from_state_dict(item)
            for item in state["device_streams"]
        ]
        obj._repeat_streams = [
            ChaCha20Rng.from_state_dict(item)
            for item in state["repeat_streams"]
        ]
        return obj

    def _normalize_counts(self, counts: Sequence[int] | None) -> list[int]:
        if counts is None:
            return list(self.channel_counts)
        normalized = [int(v) for v in counts]
        if len(normalized) != len(self.devices):
            raise ValueError("channel_counts length must match devices")
        if any(v < 0 for v in normalized):
            raise ValueError("channel_counts must be non-negative")
        return normalized
