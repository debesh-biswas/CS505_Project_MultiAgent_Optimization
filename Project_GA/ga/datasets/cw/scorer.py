"""LLM-as-judge scorer for Creative Writing (CW) outputs.

Zero-shot prompt, 5 independent calls per output, scores averaged (1–10 scale).
Uses the same NVIDIA NIM OpenAI-compatible API as the rest of the pipeline.
"""

from __future__ import annotations

import re
from typing import List

from ._backoff import call_with_backoff

_JUDGE_SYSTEM = (
    "You are an expert creative writing evaluator. "
    "You give concise, fair numeric scores."
)

_JUDGE_TEMPLATE = """\
You are evaluating a piece of creative writing.

The writer was given these 4 sentences to incorporate naturally into a coherent passage \
(one sentence to end each of 4 paragraphs):
{sentences}

Here is the generated passage:
{text}

Rate the quality of this passage on a scale of 1 to 10, considering:
- Coherence and narrative flow across paragraphs
- How naturally each of the 4 sentences is incorporated
- Overall creativity and writing quality

Respond with only a single integer from 1 to 10. Do not include any explanation."""


def _parse_score(response: str, fallback: float = 0.0) -> float:
    """Extract the first integer 1–10 from the response string."""
    matches = re.findall(r"\b([1-9]|10)\b", response.strip())
    if matches:
        return float(matches[0])
    return fallback


def cw_llm_score(
    generated_text: str,
    sentences: List[str],
    model: str,
    api_key: str,
    base_url: str,
    n_samples: int = 5,
) -> float:
    """
    Score a generated CW passage using an LLM judge.

    Calls the judge ``n_samples`` times independently (zero-shot) and returns
    the mean score in [1, 10]. Parse failures fall back to 5.0 (neutral).
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    sentences_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sentences))
    prompt = _JUDGE_TEMPLATE.format(sentences=sentences_block, text=generated_text)

    def _single_judge_call() -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=8,
        )
        return resp.choices[0].message.content or ""

    scores: List[float] = []
    for _ in range(n_samples):
        try:
            content = call_with_backoff(_single_judge_call)
            scores.append(_parse_score(content))
        except Exception as e:
            print(f"  [CW judge] call failed after retries: {e}")
            scores.append(0.0)

    mean = sum(scores) / len(scores) if scores else 5.0
    return mean


def cw_llm_score_local(
    generated_text: str,
    sentences: List[str],
    model: str,
    n_samples: int = 5,
) -> float:
    """Score using the local vLLM offline engine (no HTTP server or API key needed)."""
    from vllm import SamplingParams
    from ga.local_llm import _get_llm, _vllm_lock

    llm = _get_llm(model)
    sentences_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sentences))
    prompt = _JUDGE_TEMPLATE.format(sentences=sentences_block, text=generated_text)
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    sampling = SamplingParams(temperature=0.7, max_tokens=8)

    scores: List[float] = []
    for _ in range(n_samples):
        try:
            with _vllm_lock:
                outputs = llm.chat(messages=messages, sampling_params=sampling)
            text = outputs[0].outputs[0].text
            scores.append(_parse_score(text))
        except Exception as e:
            print(f"  [CW judge local] call failed: {e}")
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 5.0
