# triton-csprng

`triton-csprng` is a small PyTorch/Triton package for counter-based random
streams on NVIDIA GPUs. It provides ChaCha20-backed CUDA tensor generation and a
few sampling primitives that are useful for cryptography-adjacent, simulation,
and FHE-style workloads.

The package exposes low-level stream and sampling building blocks without
depending on any downstream library's RNG API.

## What is implemented

- ChaCha20 block generation in Triton.
- Explicit key / nonce / counter stream state.
- Raw `uint32(...)` and `bytes(...)` APIs returning ordinary CUDA tensors.
- Bounded integer sampling with scalar or per-channel bounds.
- Centered integer discrete Gaussian sampling from a 128-bit half-plane CDT.
- Stochastic rounding for CUDA floating tensors.
- `RnsRandomStreams`, a convenience manager for RNS-like layouts with:
  - independent streams per device for non-repeated channels;
  - repeated channels that reproduce the same values across devices;
  - state-dict roundtrip for deterministic continuation.
- No C++/CUDA extension or `torch.ops` registration step.

## Installation

For ordinary package use after a PyPI release:

```bash
python -m pip install triton-csprng
```

For development:

```bash
git clone git@github.com:visualDust/triton-csprng.git
cd triton-csprng
python -m pip install -e ".[dev]"
```

Runtime dependencies are PyTorch, Triton, and `mpmath` for high-precision CDT
construction. The current implementation is CUDA-only because Triton kernels
require CUDA tensors. In production-like CUDA environments, install the
PyTorch/Triton build that matches the target CUDA stack first, then install this
package.

## Quick start

```python
import torch
from triton_csprng import ChaCha20Rng

rng = ChaCha20Rng(
    key=list(range(8)),      # 8 little-endian uint32 words = 256 bits
    nonce=[123, 456],        # 2 little-endian uint32 words = 64 bits
    counter=0,
    device="cuda:0",
)

words = rng.uint32((1024,))
raw = rng.bytes((4096,))
mod_q = rng.randint([17, 257], (2, 1024))
gauss = rng.discrete_gaussian((4, 1024), sigma=3.2)
rounded = rng.stochastic_round(torch.randn(1024, device="cuda:0"))
```

Every result above is a normal PyTorch CUDA tensor. Triton kernels are launched
directly from Python with PyTorch tensor pointers, so callers can pass outputs
straight into ordinary PyTorch code.

## Stream semantics

`ChaCha20Rng` is a deterministic counter-based stream:

```python
from triton_csprng import ChaCha20Rng

rng1 = ChaCha20Rng(key=list(range(8)), nonce=[1, 2], counter=9)
rng2 = ChaCha20Rng(key=list(range(8)), nonce=[1, 2], counter=9)

assert torch.equal(rng1.uint32((3, 7)), rng2.uint32((3, 7)))
```

The stream buffers unused bytes internally, so chunked reads are equivalent to a
single larger read:

```python
one_shot = ChaCha20Rng(key=list(range(8)), nonce=[5, 6])
chunked = ChaCha20Rng(key=list(range(8)), nonce=[5, 6])

expected = one_shot.uint32(40)
got = torch.cat([chunked.uint32(17), chunked.uint32(23)])
assert torch.equal(got, expected)
```

Stream state can be checkpointed:

```python
state = rng1.state_dict()
restored = ChaCha20Rng.from_state_dict(state)
```

## Sampling APIs

### Bounded integers

```python
x = rng.randint(17, (4, 1024))
y = rng.randint([17, 257, 65537], (3, 1024))
```

The output dtype is `torch.int64`; bounds therefore must fit in signed int64.
Multiple bounds are interpreted as leading channels. Internally the sampler
consumes a 128-bit ChaCha word `U` and returns `floor(bound * U / 2**128)`. This
has no retry or fallback branch. Over the finite 128-bit source domain, output
bucket counts differ by at most one, so the statistical distance from the ideal
uniform distribution is bounded by roughly `bound / 2**128`.

### Discrete Gaussian

```python
e = rng.discrete_gaussian((8, 32768), sigma=3.2)
```

The sampler builds a 128-bit half-plane cumulative distribution table using
`mpmath` precision `2 * security_bits`, chooses
`num_sampling_points = 2**ceil(log2(6*sigma))`, reserves one random bit for the
sign, and folds the remaining truncated tail into the last bucket. This keeps the
CUDA path compact and constant-shape while making the finite table construction
explicit and testable.

### Stochastic rounding

```python
rounded = rng.stochastic_round(values)
```

`values` must be a CUDA floating tensor on the same device as the RNG stream.
The result is `torch.int64` with Bernoulli rounding by the fractional part of
`abs(values)`, then sign restoration.

## RNS-style stream manager

`RnsRandomStreams` helps express layouts where each GPU gets independent
non-repeated channels, while repeated channels are generated from matching
streams on every GPU.

```python
from triton_csprng import RnsRandomStreams

streams = RnsRandomStreams(
    num_coeffs=32768,
    channel_counts=[8, 8],
    repeated_channels=2,
    devices=["cuda:0", "cuda:1"],
    key=list(range(8)),
    nonce=[1, 2],
)

u32 = streams.uint32_channels()
gauss = streams.discrete_gaussian_channels(sigma=3.2)
ints = streams.randint_channels([
    [17] * 8 + [257] * 2,
    [19] * 8 + [257] * 2,
])
```

For each returned list item:

```text
shape = [non_repeated_channels + repeated_channels, num_coeffs]
```

The repeated tail channels are reproducible across devices when their bounds and
distribution parameters match.

## Why there is no torch op wrapper

A Triton kernel can be launched directly with PyTorch CUDA tensors:

```python
out = torch.empty_like(x)
_kernel[grid](x, out, ...)
```

That is what this package does. A `torch.ops.*` custom op is unnecessary for
normal Python/PyTorch integration and would reintroduce dispatcher/wrapper
maintenance. A `torch.library` wrapper can still be added later if a downstream
project needs formal fake-tensor or `torch.compile` dispatcher integration.

## Developer checks

Install the optional development tools and pre-commit hooks:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Run the same checks manually:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest tests -q
```

## Validation

Current local validation:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pre_commit run --all-files
python -m pytest tests -q
```

The tests cover:

- ChaCha20 Triton output against a Python reference implementation;
- non-multiple block counts;
- stream determinism and chunking behavior;
- state-dict restore;
- bounded integer range and multiply-high mapping checks;
- difficult bounds and distribution sanity;
- half-plane CDT table shape/range checks;
- rough discrete Gaussian moments and symmetry;
- stochastic-rounding determinism and integer cases;
- RNS repeated-channel equality across two GPUs.
