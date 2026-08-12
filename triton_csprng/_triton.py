from __future__ import annotations


class _UnavailableTriton:
    """Keep CUDA kernel definitions importable without a Triton installation."""

    def jit(self, function):
        return function

    def __getattr__(self, name: str):
        raise RuntimeError(
            "CUDA random sampling requires Triton; install triton-csprng[cuda]"
        )


class _UnavailableTritonLanguage:
    constexpr = object()

    def __getattr__(self, name: str):
        raise RuntimeError(
            "CUDA random sampling requires Triton; install triton-csprng[cuda]"
        )


try:
    import triton as triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = _UnavailableTriton()
    tl = _UnavailableTritonLanguage()
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True


def require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "CUDA random sampling requires Triton; install triton-csprng[cuda]"
        )
