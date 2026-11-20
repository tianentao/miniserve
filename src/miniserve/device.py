"""Device selection: the single place in MiniServe that decides where tensors live.

Only hardware-facing code (scripts, ModelRunner, benchmarks) should import this.
Scheduler and BlockManager must stay hardware-free and never import it.
"""

import torch

VALID_DEVICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(requested: str = "auto") -> torch.device:
    """Turn a user's device choice into a torch.device that works on this machine.

    Rules (decided in M0 Task 0.2):
      - Input is normalized: " MPS " -> "mps".
      - "auto" picks the first available of: cuda -> mps -> cpu.
      - An explicit device that is unavailable raises RuntimeError
        (never silently fall back: that would corrupt benchmark labels).
      - Anything not in VALID_DEVICES raises ValueError.
      - "cpu" always succeeds.
    """
    requested = requested.strip().lower()
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"Invalid device '{requested}', must be one of {VALID_DEVICES}"
        )

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda' is not available on this machine.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested device 'mps' is not available on this machine.")

    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    """Block until all queued work on `device` has finished.

    GPU calls return before the GPU is done, so benchmarks must call this
    before reading the clock. No-op on CPU.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        # MPS is asynchronous, like CUDA: ops are queued and run later.
        torch.mps.synchronize()
    # CPU: ops finish before they return, so there is nothing to wait for.
