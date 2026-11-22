"""Benchmark the naive (no KV cache) baseline across devices and prompt lengths.

Methodology:
  - Fixed token counts: EOS is ignored, so every run generates exactly N tokens.
  - Warmup runs are discarded (first GPU calls pay one-time setup costs).
  - Each config runs several trials; we report the median and the min/max spread.
  - Results + environment (versions, chip, threads, git commit) saved as JSON.

Example:
    uv run python benchmarks/bench_baseline.py
    uv run python benchmarks/bench_baseline.py --devices mps --prompt-lens 8 256 --trials 3
"""

import argparse
import json
import platform
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniserve.baseline import naive_greedy_generate
from miniserve.device import resolve_device

RESULTS_DIR = Path(__file__).parent / "results"
SOURCE_TEXT = (
    "The history of computing is a story of people building machines to extend "
    "the reach of human thought, from mechanical calculators to modern processors. "
)


def make_prompt_ids(tokenizer, length: int) -> list[int]:
    """A prompt of EXACTLY `length` tokens (repeat source text, then truncate)."""
    ids = tokenizer.encode(SOURCE_TEXT)
    return (ids * (length // len(ids) + 1))[:length]


def environment() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    dirty = run(["git", "status", "--porcelain"]) not in ("", "unknown")
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": run(["git", "rev-parse", "--short", "HEAD"])
        + ("-dirty" if dirty else ""),
        "chip": run(["sysctl", "-n", "machdep.cpu.brand_string"])
        or platform.processor(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cpu_threads": torch.get_num_threads(),
        "dtype": "float32",
    }


def bench_config(model, prompt_ids, new_tokens, ctx, warmup, trials) -> dict:
    for _ in range(warmup):
        naive_greedy_generate(model, prompt_ids, new_tokens, None, ctx)

    tok_per_s, first_ms, last_ms = [], [], []
    for _ in range(trials):
        r = naive_greedy_generate(model, prompt_ids, new_tokens, None, ctx)
        tok_per_s.append(len(r.token_ids) / sum(r.step_times_s))
        first_ms.append(r.step_times_s[0] * 1000)
        last_ms.append(r.step_times_s[-1] * 1000)

    return {
        "tok_per_s_median": statistics.median(tok_per_s),
        "tok_per_s_min": min(tok_per_s),
        "tok_per_s_max": max(tok_per_s),
        "first_step_ms_median": statistics.median(first_ms),
        "last_step_ms_median": statistics.median(last_ms),
        "tok_per_s_all": tok_per_s,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[8, 128, 512])
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--model", default="openai-community/gpt2")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    env = environment()
    print(
        f"{env['chip']} | torch {env['torch']} | {env['cpu_threads']} CPU threads | commit {env['git_commit']}"
    )
    print(
        f"{args.new_tokens} new tokens, {args.warmup} warmup + {args.trials} trials per config\n"
    )
    print(
        f"{'device':<6} {'prompt':>6} | {'tok/s median':>12} {'(min - max)':>15} | {'first step':>10} {'last step':>10}"
    )
    print("-" * 72)

    results = []
    for dev_name in args.devices:
        device = resolve_device(dev_name)
        model = (
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
            .eval()
            .to(device)
        )
        ctx = model.config.n_positions
        for plen in args.prompt_lens:
            r = bench_config(
                model,
                make_prompt_ids(tokenizer, plen),
                args.new_tokens,
                ctx,
                args.warmup,
                args.trials,
            )
            results.append(
                {
                    "device": str(device),
                    "prompt_len": plen,
                    "new_tokens": args.new_tokens,
                    **r,
                }
            )
            spread = f"({r['tok_per_s_min']:.1f} - {r['tok_per_s_max']:.1f})"
            print(
                f"{device!s:<6} {plen:>6} | {r['tok_per_s_median']:>12.1f} {spread:>15} | "
                f"{r['first_step_ms_median']:>8.1f}ms {r['last_step_ms_median']:>8.1f}ms"
            )
        del model

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"baseline_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(
            {"environment": env, "args": vars(args), "results": results}, indent=2
        )
    )
    print(
        f"\nsaved: {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}"
    )


if __name__ == "__main__":
    main()
