"""Naive greedy generation: the M0 correctness baseline.

No KV cache. Every step re-runs the model on the ENTIRE sequence so far.
This is deliberately slow and obviously correct: every faster version we build
later (KV cache in M1, paged attention in M4) must produce identical tokens.

Library vs. ours:
  - Hugging Face provides the model's forward pass (replaced by our own in M1).
  - The generation loop, stopping rules, and timing below are ours.
"""

import time
from dataclasses import dataclass

import torch


@dataclass
class GenerationResult:
    token_ids: list[int]  # ONLY the newly generated tokens (prompt not included)
    step_times_s: list[float]  # wall-clock seconds per step; len == len(token_ids)


@torch.inference_mode()  # no gradients needed: saves memory and time
def naive_greedy_generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None,
    max_context_len: int,
) -> GenerationResult:
    """Generate up to `max_new_tokens` tokens greedily, recomputing everything each step.

    Args:
        model: a causal LM in eval mode, already on its device.
            model(input_ids=x, use_cache=False).logits has shape [batch, seq_len, vocab].
        prompt_ids: tokenized prompt. Must be non-empty.
        max_new_tokens: stop after this many tokens. 0 is allowed (returns nothing).
        eos_token_id: stop right after generating this token (it IS included in
            the output). None means never stop early.
        max_context_len: the model's position limit (1024 for GPT-2).

    Invariants:
        - len(result.token_ids) <= max_new_tokens
        - if eos_token_id appears in result.token_ids, it is the last element
        - len(result.token_ids) == len(result.step_times_s)

    Raises:
        ValueError: empty prompt, negative max_new_tokens, or
            len(prompt_ids) + max_new_tokens > max_context_len.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    total_len = len(prompt_ids) + max_new_tokens
    if total_len > max_context_len:
        raise ValueError(
            f"prompt ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) = "
            f"{total_len} exceeds max_context_len ({max_context_len})"
        )

    device = next(model.parameters()).device

    token_ids: list[int] = []
    step_times_s: list[float] = []
    for _ in range(max_new_tokens):
        # Timer covers the whole step: host->device copy, forward pass, argmax.
        start_time = time.perf_counter()

        # Build the input tensor: ALL tokens so far (no KV cache), shape [1, seq_len]
        input_ids = torch.tensor([prompt_ids + token_ids], device=device)

        # Run the model; logits shape [1, seq_len, vocab]
        logits = model(input_ids, use_cache=False).logits

        # Greedy: argmax of the LAST position. .item() copies the result to the
        # CPU, which forces the GPU to finish first, so it acts as our sync point.
        next_token_id = logits[0, -1].argmax().item()

        step_time = time.perf_counter() - start_time

        # Record the token and step time
        token_ids.append(next_token_id)
        step_times_s.append(step_time)

        # Stop if we hit the EOS token
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return GenerationResult(token_ids=token_ids, step_times_s=step_times_s)
