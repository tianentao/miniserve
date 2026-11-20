"""Tests for miniserve.device.

Most tests fake the hardware with monkeypatch, so every branch of the logic
runs on any machine (a Mac, a CUDA box, or a CPU-only CI runner).
"""

import pytest
import torch

from miniserve.device import resolve_device, synchronize


@pytest.fixture
def fake_hw(monkeypatch):
    """Return a function that pretends CUDA / MPS are (un)available.

    monkeypatch swaps the real torch functions for fakes during one test,
    then automatically restores the originals when the test ends.
    """

    def _set(cuda: bool, mps: bool) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    return _set


# ---- resolve_device: "auto" priority order (cuda -> mps -> cpu) ----


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),  # both present: CUDA wins
        (True, False, "cuda"),
        (False, True, "mps"),  # a Mac
        (False, False, "cpu"),  # a CPU-only CI runner
    ],
)
def test_auto_priority(fake_hw, cuda, mps, expected):
    fake_hw(cuda=cuda, mps=mps)
    assert resolve_device("auto") == torch.device(expected)


def test_default_argument_is_auto(fake_hw):
    fake_hw(cuda=False, mps=False)
    assert resolve_device() == torch.device("cpu")


# ---- resolve_device: explicit choices ----


def test_cpu_always_works(fake_hw):
    fake_hw(cuda=False, mps=False)
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("name", ["cuda", "mps"])
def test_explicit_available_device(fake_hw, name):
    fake_hw(cuda=True, mps=True)
    assert resolve_device(name) == torch.device(name)


@pytest.mark.parametrize("name", ["cuda", "mps"])
def test_explicit_unavailable_device_raises(fake_hw, name):
    """Decision 1: never silently fall back to CPU."""
    fake_hw(cuda=False, mps=False)
    with pytest.raises(RuntimeError, match=name):
        resolve_device(name)


# ---- resolve_device: input handling ----


@pytest.mark.parametrize("raw", [" MPS ", "Mps", "mps\n"])
def test_input_is_normalized(fake_hw, raw):
    """Decision 3: case and surrounding whitespace don't matter."""
    fake_hw(cuda=False, mps=True)
    assert resolve_device(raw) == torch.device("mps")


@pytest.mark.parametrize("bad", ["gpu", "", "cuda:0", "metal"])
def test_invalid_device_raises_value_error(bad):
    # "cuda:0" is rejected on purpose: we don't support device indices (yet).
    with pytest.raises(ValueError, match="must be one of"):
        resolve_device(bad)


# ---- synchronize ----


@pytest.fixture
def sync_calls(monkeypatch):
    """Replace the real sync functions with recorders, so we can check which
    one synchronize() calls without needing that hardware."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append("cuda"))
    monkeypatch.setattr(torch.mps, "synchronize", lambda *a, **k: calls.append("mps"))
    return calls


@pytest.mark.parametrize(
    ("device", "expected_calls"),
    [
        ("cpu", []),  # CPU: nothing to wait for
        ("mps", ["mps"]),
        ("cuda", ["cuda"]),
    ],
)
def test_synchronize_dispatches_by_device_type(sync_calls, device, expected_calls):
    synchronize(torch.device(device))
    assert sync_calls == expected_calls


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_synchronize_real_mps():
    """One real-hardware check: the result is ready and correct after syncing."""
    device = torch.device("mps")
    x = torch.ones(512, 512, device=device)
    y = x @ x
    synchronize(device)
    assert y[0, 0].item() == 512.0
