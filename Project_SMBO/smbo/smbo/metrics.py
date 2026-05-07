"""Compatibility metrics module for TravelPlanner evaluation.

Provides the same symbols expected by evaluators that import:
    from .metrics import aggregate_travelplanner_batch, metrics_row_tp_from_boxes
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def metrics_row_tp_from_boxes(
    delivered: bool,
    commonsense_info_box: Optional[Dict[str, Any]],
    hard_info_box: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one per-sample metrics row from constraint boxes."""
    row: Dict[str, Any] = {
        "delivered": delivered,
        "cs_pass": 0,
        "cs_total": 0,
        "hd_pass": 0,
        "hd_total": 0,
        "commonsense_macro_ok": False,
        "hard_macro_ok": False,
        "final_pass": False,
    }
    if not delivered or not commonsense_info_box:
        return row

    cs_macro = True
    for _k, v in commonsense_info_box.items():
        if v[0] is not None:
            row["cs_total"] += 1
            if v[0] is True:
                row["cs_pass"] += 1
            else:
                cs_macro = False
    row["commonsense_macro_ok"] = cs_macro

    hard_box = hard_info_box or {}
    hard_macro = bool(hard_box)
    for _k, v in hard_box.items():
        if v[0] is not None:
            row["hd_total"] += 1
            if v[0] is True:
                row["hd_pass"] += 1
            else:
                hard_macro = False
    row["hard_macro_ok"] = hard_macro

    row["final_pass"] = bool(cs_macro and hard_macro)
    return row


def aggregate_travelplanner_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-sample rows into batch rates."""
    n = len(rows)
    if n == 0:
        return {
            "n_samples": 0,
            "delivery_rate": None,
            "commonsense_constraint_micro_pass_rate": None,
            "commonsense_constraint_macro_pass_rate": None,
            "hard_constraint_micro_pass_rate": None,
            "hard_constraint_macro_pass_rate": None,
            "final_pass_rate": None,
            "note": "empty batch",
        }

    n_del = sum(1 for r in rows if r["delivered"])
    delivery_rate = n_del / n

    cs_num = sum(r["cs_pass"] for r in rows)
    cs_den = sum(r["cs_total"] for r in rows)
    cs_micro = (cs_num / cs_den) if cs_den > 0 else None

    cs_macro_ct = sum(1 for r in rows if r["commonsense_macro_ok"])
    cs_macro = cs_macro_ct / n

    hd_num = sum(r["hd_pass"] for r in rows)
    hd_den = sum(r["hd_total"] for r in rows)
    hd_micro = (hd_num / hd_den) if hd_den > 0 else None

    hd_macro_ct = sum(1 for r in rows if r["hard_macro_ok"])
    hd_macro = hd_macro_ct / n

    final_ct = sum(1 for r in rows if r["final_pass"])
    final_pass = final_ct / n

    return {
        "n_samples": n,
        "n_delivered": n_del,
        "delivery_rate": delivery_rate,
        "commonsense_constraint_micro_pass_rate": cs_micro,
        "commonsense_constraint_macro_pass_rate": cs_macro,
        "hard_constraint_micro_pass_rate": hd_micro,
        "hard_constraint_macro_pass_rate": hd_macro,
        "final_pass_rate": final_pass,
        "note": "Batch-only rates.",
    }
