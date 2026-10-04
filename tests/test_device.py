import pytest
import torch

from minillm.device import select_device


def test_auto_picks_an_available_device():
    info = select_device("auto")
    x = torch.ones(4, dtype=info.dtype, device=info.device)
    assert x.sum().item() == 4


def test_cpu_is_always_available():
    assert select_device("cpu").device.type == "cpu"


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        select_device("tpu")
