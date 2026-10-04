"""Device and dtype selection, so the same code runs on the MacBook (MPS), CUDA boxes and CPU."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    dtype: torch.dtype
    name: str

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def _mps_supports_bf16() -> bool:
    # bf16 on MPS needs macOS 14+; probe instead of parsing OS versions.
    try:
        torch.zeros(1, dtype=torch.bfloat16, device="mps")
        return True
    except (RuntimeError, TypeError):
        return False


def select_device(requested: str = "auto") -> DeviceInfo:
    """Pick a device. `requested` is one of: auto, cuda, mps, cpu."""
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return DeviceInfo(torch.device("cuda"), torch.bfloat16, torch.cuda.get_device_name(0))
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        dtype = torch.bfloat16 if _mps_supports_bf16() else torch.float16
        return DeviceInfo(torch.device("mps"), dtype, "Apple MPS")
    if requested == "cpu":
        return DeviceInfo(torch.device("cpu"), torch.float32, "CPU")
    raise ValueError(f"unknown device: {requested!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the device and dtype minillm will use.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    info = select_device(parser.parse_args().device)
    print(f"device={info.device} dtype={info.dtype} name={info.name}")


if __name__ == "__main__":
    main()
