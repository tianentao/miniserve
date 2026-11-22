"""Tests for miniserve.baseline.naive_greedy_generate.

Two layers:
  1. Logic tests with a tiny scripted fake model: fast, exact, no download.
     They check stopping rules, validation, and the "recompute everything" behavior.
  2. Golden tests with real GPT-2: our loop must produce EXACTLY the same tokens
     as Hugging Face's model.generate(), an independent implementation.
"""

from types import SimpleNamespace

import pytest
import torch

from miniserve.baseline import naive_greedy_generate

# ============================================================================
# Layer 1: logic tests with a scripted fake model
# ============================================================================

EOS = 9


class ScriptedModel(torch.nn.Module):
    """A fake LM that emits a fixed sequence of tokens, one per forward call.

    It also records every input it receives, so tests can check exactly what
    the generation loop fed it.
    """

    def __init__(self, script: list[int], vocab_size: int = 10):
        super().__init__()
        # One dummy parameter so `next(model.parameters()).device` works.
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.script = script
        self.vocab_size = vocab_size
        self.inputs_seen: list[list[int]] = []

    def forward(self, input_ids, use_cache=False):
        assert use_cache is False, "baseline must not use a KV cache"
        assert input_ids.shape[0] == 1, "baseline runs a batch of exactly 1"
        self.inputs_seen.append(input_ids[0].tolist())
        next_token = self.script[len(self.inputs_seen) - 1]
        logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
        logits[0, -1, next_token] = (
            1.0  # make `next_token` the argmax at the last position
        )
        return SimpleNamespace(logits=logits)


def test_stops_at_eos_and_includes_it():
    model = ScriptedModel(script=[3, 4, EOS, 5, 6])
    result = naive_greedy_generate(model, [1, 2], 10, EOS, max_context_len=100)
    assert result.token_ids == [3, 4, EOS]  # 5 and 6 never generated
    assert len(model.inputs_seen) == 3  # model not called after EOS


def test_eos_none_never_stops_early():
    model = ScriptedModel(script=[EOS, EOS, EOS])
    result = naive_greedy_generate(model, [1], 3, None, max_context_len=100)
    assert result.token_ids == [EOS, EOS, EOS]


def test_stops_at_max_new_tokens():
    model = ScriptedModel(script=[3, 4, 5, 6, 7])
    result = naive_greedy_generate(model, [1], 2, EOS, max_context_len=100)
    assert result.token_ids == [3, 4]


def test_each_step_recomputes_the_full_sequence():
    """The defining behavior of the no-KV-cache baseline: every step re-feeds
    prompt + everything generated so far."""
    model = ScriptedModel(script=[3, 4, 5])
    naive_greedy_generate(model, [1, 2], 3, EOS, max_context_len=100)
    assert model.inputs_seen == [
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
    ]


def test_one_step_time_per_token():
    model = ScriptedModel(script=[3, 4, 5])
    result = naive_greedy_generate(model, [1], 3, EOS, max_context_len=100)
    assert len(result.step_times_s) == len(result.token_ids)
    assert all(t > 0 for t in result.step_times_s)


def test_zero_new_tokens_never_calls_model():
    model = ScriptedModel(script=[])
    result = naive_greedy_generate(model, [1, 2], 0, EOS, max_context_len=100)
    assert result.token_ids == [] and result.step_times_s == []
    assert model.inputs_seen == []


def test_exactly_at_context_limit_is_allowed():
    """Off-by-one guard: prompt + max_new_tokens == max_context_len must work."""
    model = ScriptedModel(script=[3, 4])
    result = naive_greedy_generate(model, [1, 2, 3], 2, EOS, max_context_len=5)
    assert result.token_ids == [3, 4]


@pytest.mark.parametrize(
    ("prompt", "max_new", "ctx", "message"),
    [
        ([], 5, 100, "non-empty"),
        ([1], -1, 100, ">= 0"),
        ([1, 2, 3], 3, 5, "exceeds max_context_len"),  # 6 > 5: one over the limit
    ],
)
def test_invalid_inputs_raise(prompt, max_new, ctx, message):
    model = ScriptedModel(script=[3] * 10)
    with pytest.raises(ValueError, match=message):
        naive_greedy_generate(model, prompt, max_new, EOS, max_context_len=ctx)
    assert model.inputs_seen == []  # validation happens BEFORE any compute


# ============================================================================
# Layer 2: golden tests against Hugging Face generate() with real GPT-2
# ============================================================================

MODEL_NAME = "openai-community/gpt2"
MAX_NEW_TOKENS = 32
PROMPTS = [
    "The capital of France is",  # factual
    "Hello",  # single-token prompt (edge case)
    "def fibonacci(n):",  # code
    "In 1492, Columbus",  # numbers + names
    "1, 2, 3, 4,",  # pattern continuation
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def cpu_model():
    """Loaded once per test module (scope="module"), not once per test."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()


def hf_greedy(
    model, prompt_ids: list[int], max_new_tokens: int, eos_id: int
) -> list[int]:
    """Reference: Hugging Face's own greedy generation (which uses its KV cache)."""
    x = torch.tensor([prompt_ids], device=model.device)
    out = model.generate(
        x,
        attention_mask=torch.ones_like(x),
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy
        num_beams=1,  # no beam search
        pad_token_id=eos_id,  # GPT-2 has no pad token; silences a warning
    )
    return out[0, len(prompt_ids) :].tolist()  # HF returns prompt + new; keep new only


@pytest.mark.parametrize("prompt", PROMPTS)
def test_matches_hf_generate_exactly_on_cpu(tokenizer, cpu_model, prompt):
    ids = tokenizer.encode(prompt)
    ours = naive_greedy_generate(
        cpu_model,
        ids,
        MAX_NEW_TOKENS,
        tokenizer.eos_token_id,
        cpu_model.config.n_positions,
    )
    expected = hf_greedy(cpu_model, ids, MAX_NEW_TOKENS, tokenizer.eos_token_id)
    assert ours.token_ids == expected


needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs Apple GPU"
)


@pytest.fixture(scope="module")
def mps_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    return model.eval().to("mps")


@needs_mps
@pytest.mark.parametrize("prompt", PROMPTS)
def test_mps_logits_close_to_cpu(tokenizer, cpu_model, mps_model, prompt):
    """CPU and MPS are NOT bit-identical (measured max diff ~3.6e-4 in M0), because
    floating-point ops run in a different order. We assert closeness, not equality."""
    x = torch.tensor([tokenizer.encode(prompt)])
    with torch.inference_mode():
        cpu_logits = cpu_model(x, use_cache=False).logits
        mps_logits = mps_model(x.to("mps"), use_cache=False).logits.cpu()
    torch.testing.assert_close(mps_logits, cpu_logits, atol=1e-3, rtol=0)


@needs_mps
@pytest.mark.parametrize("prompt", PROMPTS)
def test_mps_tokens_match_cpu(tokenizer, cpu_model, mps_model, prompt):
    """Despite tiny logit differences, greedy tokens agree for these prompts.
    If this ever fails, check for a near-tie between the top-2 logits before
    assuming a bug: that would be a numerics effect, not a logic error."""
    ids = tokenizer.encode(prompt)
    eos, ctx = tokenizer.eos_token_id, cpu_model.config.n_positions
    on_cpu = naive_greedy_generate(cpu_model, ids, MAX_NEW_TOKENS, eos, ctx)
    on_mps = naive_greedy_generate(mps_model, ids, MAX_NEW_TOKENS, eos, ctx)
    assert on_mps.token_ids == on_cpu.token_ids
