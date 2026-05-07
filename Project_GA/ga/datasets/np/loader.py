"""Load Natural Plan batches from Hugging Face (tuandunghcmut/natural-plan-benchmark)."""

from __future__ import annotations

import json
from typing import Any, List, Literal

NpKind = Literal["trip", "meeting", "calendar"]

_SUBSET = {
    "trip": "normalized_trip_planning",
    "meeting": "normalized_meeting_planning",
    "calendar": "normalized_calendar_scheduling",
}


def load_np_batch(
    kind: NpKind,
    max_samples: int = 0,
    seed: int = 0,
) -> List[dict[str, Any]]:
    """
    Return a list of unified sample dicts for ``evaluate_mas``.

    ``max_samples=0`` (default) loads the full test split.
    ``seed`` fixes the Hugging Face shuffle (use the same value as ``batch_seed`` in
    ``main_ga.run_ga`` for reproducible NP mini-batches).

    Fields per kind:
      - trip: task_type, prompt, cities, durations (``**`` strings), cities_raw, durations_raw
      - meeting: task_type, prompt, golden_plan_text, constraints_json, dist_matrix_json
      - calendar: task_type, prompt, golden_plan_text, golden_day, golden_start_hour, golden_end_hour
    """
    from datasets import load_dataset

    subset = _SUBSET[kind]
    ds = load_dataset("tuandunghcmut/natural-plan-benchmark", subset, split="test")
    ds = ds.shuffle(seed=seed)
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    rows: List[dict[str, Any]] = []
    for ex in ds:
        row: dict[str, Any] = {
            "task_type": kind,
            "prompt": ex["prompt_5shot"],
        }
        if kind == "trip":
            cities_raw = ex["cities_raw"]
            durations_raw = ex["durations_raw"]
            row["cities"] = cities_raw
            row["durations"] = durations_raw
            row["cities_raw"] = cities_raw
            row["durations_raw"] = durations_raw
        elif kind == "meeting":
            row["golden_plan_text"] = ex["golden_plan_text"]
            row["constraints_json"] = ex["constraints_json"]
            row["dist_matrix_json"] = ex["dist_matrix_json"]
            # Pre-parse for callers that want Python objects
            row["constraints_rows"] = json.loads(ex["constraints_json"])
            row["dist_matrix"] = json.loads(ex["dist_matrix_json"])
        else:
            row["golden_plan_text"] = ex["golden_plan_text"]
            row["golden_day"] = ex["golden_day"]
            row["golden_start_hour"] = ex["golden_start_hour"]
            row["golden_end_hour"] = ex["golden_end_hour"]
        rows.append(row)
    return rows


def load_np_batch_mixed(
    n_per_kind: int = 1,
    seed: int = 0,
) -> List[dict[str, Any]]:
    """
    Return a shuffled batch with ``n_per_kind`` samples from each of trip, meeting,
    and calendar (total = n_per_kind × 3).

    All samples carry a ``task_type`` field so the evaluator dispatches scoring
    correctly without relying on the ``np_kind`` fallback parameter.
    """
    import random as _random
    rows: List[dict[str, Any]] = []
    for kind in ("trip", "meeting", "calendar"):
        rows.extend(load_np_batch(kind, n_per_kind, seed))
    _random.Random(seed).shuffle(rows)
    return rows
