"""Run the M0 naive baseline from the command line.

Example:
    uv run python scripts/baseline_generate.py --prompt "The capital of France is" --device mps

Plumbing only: argument parsing, model loading, and printing. The generation
loop itself lives in miniserve.baseline so tests can import it.
"""

import argparse
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniserve.baseline import naive_greedy_generate
from miniserve.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Naive greedy generation (no KV cache)."
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--model", default="openai-community/gpt2")
    args = parser.parse_args()

    device = resolve_device(args.device)

    # Library: tokenizer + pretrained weights. float32 for correctness work.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = model.eval().to(device)

    prompt_ids = tokenizer.encode(args.prompt)
    max_context_len = model.config.n_positions

    # Warmup: MPS pays a one-time setup cost for EVERY new input shape, and the
    # naive loop sees a new sequence length each step. Measured in M0: a cold
    # MPS run is ~2x slower than a warm one. So warm up over the exact same
    # shapes we are about to measure (EOS ignored so all lengths are covered).
    naive_greedy_generate(model, prompt_ids, args.max_new_tokens, None, max_context_len)

    result = naive_greedy_generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        tokenizer.eos_token_id,
        max_context_len,
    )

    n = len(result.token_ids)
    total_s = sum(result.step_times_s)
    hit_eos = n > 0 and result.token_ids[-1] == tokenizer.eos_token_id

    print(
        f"model   : {args.model}  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)"
    )
    print(f"device  : {device}")
    print(f"prompt  : {args.prompt!r}  ({len(prompt_ids)} tokens)")
    print(f"output  : {tokenizer.decode(result.token_ids)!r}")
    print(
        f"stopped : {'EOS token' if hit_eos else 'max_new_tokens'}  ({n} tokens generated)"
    )
    if n == 0:
        return
    ms = [t * 1000 for t in result.step_times_s]
    print(f"total   : {total_s * 1000:.0f} ms  ->  {n / total_s:.1f} tokens/s")
    print(
        f"per step: first {ms[0]:.1f} ms | median {statistics.median(ms):.1f} ms | last {ms[-1]:.1f} ms"
        f"  (seq len {len(prompt_ids) + 1} -> {len(prompt_ids) + n})"
    )


if __name__ == "__main__":
    main()
