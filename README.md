# MiniServe

A lightweight LLM inference serving engine built from scratch in PyTorch, to learn
the core mechanisms behind systems like [vLLM](https://github.com/vllm-project/vllm):
KV caching, block-based memory management, continuous batching, and paged attention.

> **Status: Milestone 0 complete.** MiniServe currently has a *naive* greedy generation
> baseline (no KV cache, no batching). It exists as a correctness oracle and a
> performance "before" picture. KV caching, paged memory, and continuous batching are
> **not implemented yet**: see the [roadmap](#roadmap).

## Architecture (target)

```
            HTTP API (M5)
                 │
        ┌─────────────────┐
        │     Engine      │  step() loop
        └────────┬────────┘
                 ▼
        ┌─────────────────┐   ┌──────────────────┐
        │   Scheduler     │──▶│  BlockManager    │   pure Python, no torch:
        │ waiting/running │   │ free list, block │   hardware-independent,
        │ queues (M3)     │   │ tables (M2)      │   unit-testable
        └────────┬────────┘   └──────────────────┘
                 │ plain ints: seq ids, block tables      ── hardware boundary ──
                 ▼
        ┌─────────────────┐   ┌──────────────────┐
        │  ModelRunner    │──▶│ Attention backend│   torch reference first,
        │  + our GPT-2    │   │ (paged, M4)      │   Triton/CUDA later
        └─────────────────┘   └──────────────────┘
```

Scheduling and memory management only handle integers and never import `torch`.
Moving to CUDA only touches code below the hardware boundary.

## What we implement vs. what libraries provide

| Component | Source |
|---|---|
| Generation loop, stopping rules, per-step timing | **MiniServe** (`miniserve/baseline.py`) |
| Device selection & device-agnostic sync | **MiniServe** (`miniserve/device.py`) |
| Benchmark methodology & harness | **MiniServe** (`benchmarks/`) |
| Tensor ops, MPS/CUDA kernels | PyTorch |
| Tokenizer, pretrained GPT-2 weights | Hugging Face `transformers` |
| GPT-2 forward pass | Hugging Face *for now*. Replaced by our own module in M1 |
| `model.generate()` | Hugging Face, **used only as a test oracle**, never for serving |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed automatically.

```bash
uv sync                                   # exact environment from uv.lock
uv run python scripts/baseline_generate.py --prompt "The capital of France is" --device auto
uv run pytest                             # 46 tests (MPS tests skip without an Apple GPU)
uv run python benchmarks/bench_baseline.py
```

## Testing

- **Logic tests** use a scripted fake model to check stopping rules, validation, and
  that the baseline recomputes the full sequence every step.
- **Golden tests**: our loop must produce *exactly* the same tokens as Hugging Face's
  `generate()` (greedy) on 5 prompts × 32 tokens. HF uses a KV cache internally and we
  don't, so agreement checks both implementations against each other.
- **Device tests** fake hardware availability with `monkeypatch`, so every branch of
  device selection runs on any machine.
- **CPU vs. MPS**: logits are *not* bit-identical (max diff ≈ 3.6e-4, from different
  floating-point operation order), so we assert `atol=1e-3` on logits and identical
  greedy tokens.

## M0 baseline benchmark

Naive greedy generation (no KV cache), GPT-2 124M, float32, 64 new tokens, EOS ignored.
2 warmup + 5 trials per config, median reported.
Apple M4, 16 GB, torch 2.15.0, 4 CPU threads, commit `b6e0ff1`.
Raw data: [`benchmarks/results/baseline_20261122_001405.json`](benchmarks/results/baseline_20261122_001405.json).

| Device | Prompt len | tok/s (median) | min – max | First step | Last step |
|---|---:|---:|---:|---:|---:|
| CPU | 8 | 44.8 | 44.3 – 44.9 | 12.8 ms | 31.0 ms |
| CPU | 128 | 20.1 | 19.9 – 20.3 | 40.9 ms | 54.9 ms |
| CPU | 512 | 7.3 | 7.3 – 7.3 | 126.4 ms | 142.6 ms |
| MPS | 8 | 81.5 | 81.2 – 81.7 | 8.1 ms | 15.6 ms |
| MPS | 128 | 38.5 | 38.0 – 38.6 | 21.4 ms | 29.4 ms |
| MPS | 512 | 12.8 | 12.8 – 12.8 | 71.5 ms | 81.0 ms |

**Findings**

1. **Per-step cost grows with sequence length.** Without a KV cache every step
   reprocesses the whole sequence; throughput falls ~6× from prompt length 8 to 512.
   This is the cost the KV cache (M1) is meant to remove.
2. **Warm MPS is ~1.6–1.9× faster than CPU** at every prompt length, with
   non-overlapping min–max ranges.
3. **MPS pays a setup cost per new input shape.** A cold MPS run measured ~2× slower
   than a warm one (41 vs. 81 tok/s); CPU showed no such effect. Because the naive loop
   sees a new sequence length every step, warmup must cover the same shapes being
   measured. These numbers are therefore *warm-shape* numbers and optimistic relative to
   serving varied prompts. (vLLM addresses a related problem on CUDA by capturing CUDA
   graphs for fixed batch sizes and padding to them.)
4. **Small inputs hit a fixed-overhead floor** (~8 ms/step on MPS at 8 tokens), mostly
   kernel launch overhead rather than math: a motivation for batching (M3).

## Repository layout

```
src/miniserve/
  device.py          resolve_device() and synchronize()
  baseline.py        naive_greedy_generate(): M0 correctness oracle
scripts/
  baseline_generate.py   CLI: generate text + per-step timing
benchmarks/
  bench_baseline.py      reproducible CPU vs. MPS benchmark
  results/               JSON results with environment metadata
tests/                   pytest suite
```

## Roadmap

- [x] **M0**: environment, device abstraction, naive baseline, golden tests, benchmark
- [ ] **M1**: our own GPT-2 module, prefill/decode split, contiguous KV cache
- [ ] **M2**: block-based KV cache: block pool, free list, block tables
- [ ] **M3**: continuous batching: request states, waiting/running queues, scheduler
- [ ] **M4**: paged attention over block tables (PyTorch reference, then optimized)
- [ ] **M5**: HTTP serving API, throughput/latency benchmarks, CUDA/Triton experiments
