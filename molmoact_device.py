"""Device-agnostic helpers for the XPU / CUDA / CPU backends.

The inference entry points (``scripts/reconstruct_from_tokens.py`` and
``experiments/libero/run_libero_eval.py``) go through this module instead of asking
``torch.cuda`` directly.

Detection order is XPU, then CUDA, then CPU, and an explicit ``device=`` argument always
wins over detection. ``torch.cuda.*`` calls raise ``AssertionError: Torch not compiled with
CUDA enabled`` on a wheel built without CUDA, so every helper checks availability first.
"""

from __future__ import annotations

from typing import Any

import torch

DeviceLike = str | torch.device | None
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu`` > ``cuda`` > ``cpu``."""
    for name in _ACCELERATORS:
        if is_available(name):
            return name
    return "cpu"


def default_device(index: int = 0) -> torch.device:
    """Device of the best available backend (``cpu`` has no index)."""
    name = default_device_type()
    return torch.device("cpu") if name == "cpu" else torch.device(name, int(index))


def resolve_device(device: DeviceLike) -> torch.device:
    """Resolve ``None`` / str / ``torch.device``. Explicit values are honoured verbatim."""
    if isinstance(device, torch.device):
        return device
    if device is None:
        return default_device()
    return torch.device(device)


def device_type(device: Any) -> str:
    """Backend type of a device, device string or tensor."""
    if isinstance(device, torch.Tensor):
        return device.device.type
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def seed_all(seed: int) -> None:
    """Seed the device RNGs of every available accelerator (the CPU seed is the caller's)."""
    if is_available("cuda"):
        torch.cuda.manual_seed_all(seed)
    if is_available("xpu"):
        torch.xpu.manual_seed_all(seed)


def empty_cache(device: DeviceLike = None) -> None:
    """Release cached allocator blocks for ``device``'s backend; no-op on CPU."""
    name = device_type(device) if device is not None else default_device_type()
    if name in _ACCELERATORS and is_available(name):
        getattr(torch, name).empty_cache()


def autocast_device_type(device: DeviceLike = None) -> str:
    """Value for ``torch.autocast(device_type=...)``: the backend of ``device``."""
    return device_type(device) if device is not None else default_device_type()


def autocast_dtype(device: DeviceLike = None) -> torch.dtype:
    """Dtype autocast uses for ``device``'s backend.

    ``torch.get_autocast_gpu_dtype()`` reports the CUDA dtype (float16) even inside an XPU
    autocast context, so read the per-backend value instead.
    """
    name = autocast_device_type(device)
    if name not in _ACCELERATORS:
        return torch.float32
    return torch.get_autocast_dtype(name)
