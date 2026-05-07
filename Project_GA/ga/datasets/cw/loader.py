"""Load Creative Writing (CW) batches from Tree of Thoughts benchmark (Yao et al., 2024).

Single file: data_100_random_text.txt — 100 lines, 4 sentences per line.
Lines 1–5   → training tasks (skipped per paper Appendix C.1).
Lines 6–100 → the 95 evaluation tasks used here.
"""

from __future__ import annotations

import random
import urllib.request
from typing import Any, List

_DATA_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm"
    "/master/src/tot/data/text/data_100_random_text.txt"
)

# Indices into the 100-line file: 0–4 = training (skip), 5–99 = eval (use)
_EVAL_START = 5


def _fetch_all_tasks() -> List[List[str]]:
    """Download the file and return all 100 tasks as lists of 4 sentences."""
    with urllib.request.urlopen(_DATA_URL, timeout=15) as resp:
        text = resp.read().decode("utf-8")
    tasks: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on ". " to separate sentences; restore trailing period
        parts = line.split(". ")
        sentences = []
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if not part.endswith("."):
                part += "."
            sentences.append(part)
        if sentences:
            tasks.append(sentences)
    return tasks


def _build_prompt(sentences: List[str]) -> str:
    joined = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sentences))
    return (
        "Write a coherent passage of 4 short paragraphs. "
        "The end sentence of each paragraph must be exactly one of the following input sentences "
        "(use all of them, each exactly once):\n"
        f"{joined}"
    )


def load_cw_train_batch(
    max_samples: int = 0,
    seed: int = 42,
) -> List[dict[str, Any]]:
    """
    Return CW sample dicts from the **training split** (lines 1–5 of the ToT file).

    The Tree of Thoughts paper reserves these 5 tasks for tuning and explicitly
    excludes them from the 95-task eval split used by ``load_cw_batch``.

    ``max_samples=0`` (default) returns all 5 training tasks.  ``seed`` controls
    shuffling when a subset is requested.

    Each dict has the same keys as ``load_cw_batch``:
      - task_id (int): 1-based line number (1–5)
      - sentences (List[str]): the input sentences
      - prompt (str): formatted MAS task prompt
    """
    all_tasks = _fetch_all_tasks()
    train_tasks = all_tasks[:_EVAL_START]   # lines 1–5 (0-indexed 0–4)

    rng = random.Random(seed)
    indices = list(range(len(train_tasks)))
    rng.shuffle(indices)
    if max_samples and max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]

    rows: List[dict[str, Any]] = []
    for idx in indices:
        sentences = train_tasks[idx]
        rows.append(
            {
                "task_id": idx + 1,   # 1-based line number (1–5)
                "sentences": sentences,
                "prompt": _build_prompt(sentences),
                "split": "train",
            }
        )
    return rows


def load_cw_batch(
    max_samples: int = 20,
    seed: int = 42,
) -> List[dict[str, Any]]:
    """
    Return a list of CW sample dicts for ``evaluate_mas``.

    Fetches the single ToT file at runtime and uses lines 6–100 (the 95 eval tasks).
    ``max_samples=0`` loads all 95; ``>0`` takes a seeded random subset.

    Each dict has keys:
      - task_id (int): 1-based line number in the original file (6–100)
      - sentences (List[str]): the input sentences
      - prompt (str): formatted MAS task prompt
    """
    all_tasks = _fetch_all_tasks()
    eval_tasks = all_tasks[_EVAL_START:]   # lines 6–100 (0-indexed 5–99)

    rng = random.Random(seed)
    indices = list(range(len(eval_tasks)))
    rng.shuffle(indices)
    if max_samples and max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]

    rows: List[dict[str, Any]] = []
    for idx in indices:
        sentences = eval_tasks[idx]
        rows.append(
            {
                "task_id": _EVAL_START + idx + 1,   # 1-based line number
                "sentences": sentences,
                "prompt": _build_prompt(sentences),
            }
        )
    return rows
