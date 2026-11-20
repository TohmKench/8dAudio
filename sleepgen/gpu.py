"""Compute device selection for the heavy synthesis (PyTorch).

Order tried with `auto`:
  1. A CUDA-style device. AMD cards show up this way with PyTorch's ROCm builds
     (Linux, and AMD's PyTorch-on-Windows builds for Radeon RX 9000 cards).
  2. DirectML (`pip install torch-directml`), which works with any DirectX 12 GPU on Windows.
  3. CPU.

Set it with `--device` on the command line or the SLEEPGEN_DEVICE environment variable.
"""

from __future__ import annotations

import os

import numpy as np

_device = None


def _torch():
    import torch

    return torch


def resolve(name: str | None = None):
    torch = _torch()
    name = (name or os.environ.get("SLEEPGEN_DEVICE") or "auto").lower()
    if name in ("auto", "gpu", "cuda", "rocm"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if name in ("cuda", "rocm"):
            raise RuntimeError("PyTorch can't see a CUDA/ROCm GPU on this machine")
    if name in ("auto", "gpu", "directml", "dml"):
        try:
            import torch_directml  # type: ignore

            return torch_directml.device()
        except ImportError:
            if name in ("directml", "dml"):
                raise RuntimeError("torch-directml is not installed (pip install torch-directml)")
    if name == "gpu":
        raise RuntimeError("no GPU backend available (install PyTorch ROCm or torch-directml)")
    if name in ("auto", "cpu"):
        return torch.device("cpu")
    raise ValueError(f"unknown device {name!r} (auto, gpu, cuda, rocm, directml, cpu)")


def set_device(name: str | None) -> None:
    global _device
    _device = resolve(name)


def device():
    global _device
    if _device is None:
        _device = resolve(None)
    return _device


def is_gpu() -> bool:
    return device().type != "cpu"


def describe() -> str:
    torch = _torch()
    d = device()
    if d.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    if d.type == "cpu":
        return f"CPU ({torch.get_num_threads()} threads)"
    return f"GPU via {d.type}"


def default_budget() -> int:
    """Individually synthesised raindrops per second. More = more real detail."""
    env = os.environ.get("SLEEPGEN_QUALITY")
    if env:
        return int(env)
    return 16000 if is_gpu() else 2500


def batch_elements() -> int:
    """How many (event x mode x sample) values to work on at once."""
    return 96_000_000 if is_gpu() else 2_000_000


def scatter_rate() -> float:
    """Additions per second when copying each event's samples into the output (index_add_).
    Measured: ~2.9e9 on a Radeon RX 9060 XT (flat, whatever the overlap), ~2.3e8 on a CPU."""
    return 2.9e9 if is_gpu() else 2.3e8


def fft_rate() -> float:
    """Samples per second through an FFT filter bank (rfft, multiply, sum, irfft), per train.
    Measured: ~4e9 on a Radeon RX 9060 XT, ~6e8 on a CPU."""
    return 4.0e9 if is_gpu() else 6.0e8


def upload(a, dtype=None):
    """A numpy array as a tensor on the compute device.

    A plain copy: it waits for the GPU work queued before it, so callers upload everything
    they need before queueing kernels. (Asynchronous copies from pinned memory crashed
    kernels with AMD's ROCm build on Windows, and pinning is slow there.)"""
    torch = _torch()
    t = torch.from_numpy(np.ascontiguousarray(a))
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device())
