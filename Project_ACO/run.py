#!/usr/bin/env python3
"""
Run TravelPlanner ACO, Natural Plan ACO, and Creative Writing ACO via the unified ``aco``
package, then log **reportable** metrics (proposal §5.2) alongside all parameters.

Requires ``NVIDIA_API_KEY`` (and optional ``NVIDIA_API_BASE``) in the environment.

Usage (from project root)::

    set NVIDIA_API_KEY=...
    python run.py                                    # default: TP + NP + CW
    python run.py --tp-only                         # TravelPlanner only
    python run.py --np-only                         # Natural Plan only
    python run.py --cw-only                         # Creative Writing only

Outputs JSON under ``runs/`` by default, and (unless ``--no-mirror-folder-logs``)
mirrors each track into ``aco_final/run_latest.json``, ``aco_np/run_latest.json``,
and ``aco_cw/run_latest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _json_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def _save_best_config(log_dir: Path, track: str, best, log: list, params: dict) -> None:
    """Print and save the best ACO config to a checkpoint file before final evaluation."""
    best_fitness = log[-1]["best"] if log else None

    print(f"\n{'#'*60}")
    print(f"  BEST {track.upper()} CONFIG (post-ACO, pre-eval)")
    print(f"  Config    : {best}")
    if best:
        for i, a in enumerate(best.agents):
            print(f"  Agent {i}   : role={a['role']}  capability={a['capability']}")
        print(f"  Links     : {best.links}")
    print(f"  Train fitness (best ever): {best_fitness}")
    print(f"{'#'*60}\n")

    stem = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = log_dir / f"{stem}_{track}_best_config.json"
    payload = _json_safe({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "best_config_str": str(best),
        "best_agents": best.agents if best else None,
        "best_links": best.links if best else None,
        "best_fitness_training": best_fitness,
        "aco_log": log,
        "aco_params": params,
    })
    ckpt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[checkpoint] Best config saved → {ckpt_path}")


def main() -> None:
    root = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description=(
            "Run TravelPlanner (TP), Natural Plan (NP), and/or Creative Writing (CW) ACO "
            "and log reportable metrics. Default: all three tracks."
        )
    )
    p.add_argument("--model", default="meta/llama-4-maverick-17b-128e-instruct")
    run_mode = p.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--tp-only",
        action="store_true",
        help="Run only the TravelPlanner (aco_final) leg.",
    )
    run_mode.add_argument(
        "--np-only",
        action="store_true",
        help="Run only the Natural Plan (aco_np) leg.",
    )
    run_mode.add_argument(
        "--cw-only",
        action="store_true",
        help="Run only the Creative Writing (aco_cw) leg.",
    )
    p.add_argument(
        "--skip-tp",
        action="store_true",
        help="Skip TravelPlanner.",
    )
    p.add_argument(
        "--skip-np",
        action="store_true",
        help="Skip Natural Plan.",
    )
    p.add_argument(
        "--skip-cw",
        action="store_true",
        help="Skip Creative Writing.",
    )
    p.add_argument("--n-ants", type=int, default=10)
    p.add_argument("--n-iterations", type=int, default=10)
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--tp-batch", type=int, default=20, help="TravelPlanner mini-batch size (used for ACO optimization when --tp-eval-batch is not set)")
    p.add_argument("--tp-split", choices=("train", "validation", "test"), default="validation")
    p.add_argument("--tp-batch-seed", type=int, default=42)
    p.add_argument("--tp-eval-batch", type=int, default=None, help="If set, evaluate the best ACO config on this many separate queries (distinct from the train batch)")
    p.add_argument("--tp-eval-split", choices=("train", "validation", "test"), default="validation", help="Split to draw the eval batch from (default: validation)")
    p.add_argument("--tp-eval-seed", type=int, default=99, help="Seed for eval batch sampling (use a different value than --tp-batch-seed to avoid overlap)")
    p.add_argument("--tp-stratify", action="store_true", help="Sample train batch with equal easy/medium/hard queries (3 per level by default, or --tp-batch//3)")
    p.add_argument("--tp-per-level", type=int, default=None, help="Queries per difficulty level when --tp-stratify is set (overrides --tp-batch//3)")
    p.add_argument("--tp-eval-stratify", action="store_true", help="Sample eval batch with equal easy/medium/hard queries")
    p.add_argument("--tp-eval-per-level", type=int, default=None, help="Queries per difficulty level for eval batch when --tp-eval-stratify is set")
    p.add_argument("--np-kind", choices=("trip", "meeting", "calendar"), default="trip")
    p.add_argument("--np-mixed", action="store_true", help="Train and evaluate on a mixed batch of all 3 NP kinds (trip + meeting + calendar). Overrides --np-kind.")
    p.add_argument("--np-per-kind", type=int, default=1, help="Samples per kind when --np-mixed is set (train batch size = np-per-kind × 3)")
    p.add_argument("--np-batch", type=int, default=160, help="NP samples per evaluation for single-kind runs (0 = full test split); paper used ~10%% of full dataset: 160 trip / 100 meeting / 100 calendar")
    p.add_argument("--np-batch-seed", type=int, default=42)
    p.add_argument("--np-eval-per-kind", type=int, default=None, help="Samples per kind for the separate NP eval batch when --np-mixed is set (eval batch = np-eval-per-kind × 3). If unset, reuses the train batch.")
    p.add_argument("--np-eval-seed", type=int, default=99, help="Seed for the separate NP eval batch when --np-mixed is set")
    p.add_argument("--cw-train", action="store_true", help="Use the 5 official ToT training tasks for ACO optimization instead of sampling from the 95 eval tasks.")
    p.add_argument("--cw-train-batch", type=int, default=0, help="How many of the 5 training tasks to use (0 = all 5). Only applies when --cw-train is set.")
    p.add_argument("--cw-train-seed", type=int, default=42, help="Seed for sampling the CW training batch when --cw-train is set.")
    p.add_argument("--cw-batch", type=int, default=20, help="CW samples per evaluation (0 = all 95 eval tasks). Used for ACO optimization when --cw-train is NOT set.")
    p.add_argument("--cw-batch-seed", type=int, default=42)
    p.add_argument("--cw-eval-batch", type=int, default=None, help="If set, evaluate the best CW config on this many separate eval-split tasks (distinct from the train batch). Only meaningful when --cw-train is set.")
    p.add_argument("--cw-eval-seed", type=int, default=99, help="Seed for the separate CW eval batch.")
    p.add_argument("--cw-judge-n-samples", type=int, default=5, help="Independent LLM judge calls per CW output (paper: 5)")
    p.add_argument("--cw-judge-model", type=str, default=None, help="NVIDIA NIM model for the CW LLM judge (default: meta/llama-3.1-8b-instruct). Always uses NIM API, even when LOCAL_LLM=1.")
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--log-dir", type=Path, default=root / "runs")
    p.add_argument(
        "--no-mirror-folder-logs",
        action="store_true",
        help="Skip writing aco_final/run_latest.json and aco_np/run_latest.json",
    )
    args = p.parse_args()

    if args.tp_only and args.skip_tp:
        p.error("--tp-only cannot be combined with --skip-tp")
    if args.np_only and args.skip_np:
        p.error("--np-only cannot be combined with --skip-np")
    if args.cw_only and args.skip_cw:
        p.error("--cw-only cannot be combined with --skip-cw")
    if args.tp_only:
        args.skip_np = True
        args.skip_cw = True
    elif args.np_only:
        args.skip_tp = True
        args.skip_cw = True
    elif args.cw_only:
        args.skip_tp = True
        args.skip_np = True

    if not (args.skip_tp and args.skip_np and args.skip_cw) and not os.environ.get("NVIDIA_API_KEY"):
        print("ERROR: Set NVIDIA_API_KEY before running (unless all tracks are skipped).", file=sys.stderr)
        sys.exit(1)

    out: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": vars(args).copy(),
        "disclaimer": (
            "TravelPlanner numbers are batch aggregates over the sampled mini-batch, "
            "not full validation/test JSONL official eval. Natural Plan uses the "
            "HF mirror + exact-match mean on that batch. Creative Writing uses the "
            "ToT benchmark (Yao et al., 2024) with LLM-as-judge scoring (1-10, 5 samples averaged)."
        ),
        "travel_planner": None,
        "natural_plan": None,
        "creative_writing": None,
    }

    args.log_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_tp:
        from aco.main_aco import run_aco
        from aco.datasets.tp.loader import load_batch as tp_load_batch
        from aco.datasets.tp.evaluation import evaluate_mas as tp_evaluate_mas

        tp_params = {
            "n_ants": args.n_ants,
            "n_iterations": args.n_iterations,
            "num_agents": args.num_agents,
            "task_name": "TravelPlanner",
            "model": args.model,
            "rho": args.rho,
        }
        best_tp, log_tp, batch_tp = run_aco(
            **tp_params,
            load_batch_fn=lambda: tp_load_batch(
                args.tp_split, args.tp_batch, args.tp_batch_seed,
                stratify=args.tp_stratify, per_level=args.tp_per_level,
            ),
            evaluate_fn=tp_evaluate_mas,
            evaluate_kwargs={},
        )
        _save_best_config(args.log_dir, "tp", best_tp, log_tp, tp_params)
        # Load a separate eval batch if requested, otherwise reuse the train batch.
        if args.tp_eval_batch is not None:
            print(f"Loading separate TP eval batch ({args.tp_eval_split}, n={args.tp_eval_batch}, seed={args.tp_eval_seed})...")
            eval_batch_tp = tp_load_batch(
                args.tp_eval_split, args.tp_eval_batch, args.tp_eval_seed,
                stratify=args.tp_eval_stratify, per_level=args.tp_eval_per_level,
            )
        else:
            eval_batch_tp = batch_tp
        report_tp = None
        if eval_batch_tp is not None:
            _, report_tp = tp_evaluate_mas(
                best_tp,
                task_name="TravelPlanner",
                model=args.model,
                evaluation_batch=eval_batch_tp,
                return_tp_report=True,
            )
        out["travel_planner"] = {
            "aco_parameters": {**tp_params, "batch_size": args.tp_batch, "tp_split": args.tp_split, "batch_seed": args.tp_batch_seed},
            "eval_parameters": {
                "eval_batch_size": args.tp_eval_batch if args.tp_eval_batch is not None else args.tp_batch,
                "eval_split": args.tp_eval_split if args.tp_eval_batch is not None else args.tp_split,
                "eval_seed": args.tp_eval_seed if args.tp_eval_batch is not None else args.tp_batch_seed,
                "separate_eval": args.tp_eval_batch is not None,
            },
            "best_fitness_last_iter": log_tp[-1]["best"] if log_tp else None,
            "best_configuration_str": str(best_tp),
            "reportable": report_tp,
        }

    if not args.skip_np:
        from aco.main_aco import run_aco
        from aco.datasets.np.loader import load_np_batch, load_np_batch_mixed
        from aco.datasets.np.evaluation import evaluate_mas as np_evaluate_mas

        np_params = {
            "n_ants": args.n_ants,
            "n_iterations": args.n_iterations,
            "num_agents": args.num_agents,
            "task_name": "NaturalPlan",
            "model": args.model,
            "rho": args.rho,
        }

        if args.np_mixed:
            np_kind_tag = "mixed"
            np_load_batch_fn = lambda: load_np_batch_mixed(args.np_per_kind, args.np_batch_seed)
        else:
            np_kind_tag = args.np_kind
            np_load_batch_fn = lambda: load_np_batch(args.np_kind, args.np_batch, args.np_batch_seed)

        best_np, log_np, batch_np = run_aco(
            **np_params,
            load_batch_fn=np_load_batch_fn,
            evaluate_fn=np_evaluate_mas,
            evaluate_kwargs={"np_kind": args.np_kind},
        )
        _save_best_config(args.log_dir, "np", best_np, log_np, {**np_params, "np_kind": np_kind_tag})

        # Load a separate NP eval batch if requested (mixed mode only), otherwise reuse train batch.
        if args.np_mixed and args.np_eval_per_kind is not None:
            print(f"Loading separate NP eval batch (mixed, n_per_kind={args.np_eval_per_kind}, seed={args.np_eval_seed})...")
            eval_batch_np = load_np_batch_mixed(args.np_eval_per_kind, args.np_eval_seed)
        else:
            eval_batch_np = batch_np

        report_np = None
        if eval_batch_np is not None:
            _, report_np = np_evaluate_mas(
                best_np,
                task_name="NaturalPlan",
                model=args.model,
                evaluation_batch=eval_batch_np,
                np_kind=args.np_kind,
                return_np_report=True,
            )

        np_aco_params: dict = {
            **np_params,
            "np_kind": np_kind_tag,
            "batch_seed": args.np_batch_seed,
        }
        if args.np_mixed:
            np_aco_params["np_per_kind"] = args.np_per_kind
            np_aco_params["batch_size"] = args.np_per_kind * 3
            np_aco_params["eval_per_kind"] = args.np_eval_per_kind
            np_aco_params["eval_seed"] = args.np_eval_seed
        else:
            np_aco_params["batch_size"] = args.np_batch

        out["natural_plan"] = {
            "aco_parameters": np_aco_params,
            "best_fitness_last_iter": log_np[-1]["best"] if log_np else None,
            "best_configuration_str": str(best_np),
            "reportable": report_np,
        }

    if not args.skip_cw:
        from aco.main_aco import run_aco
        from aco.datasets.cw.loader import load_cw_batch, load_cw_train_batch
        from aco.datasets.cw.evaluation import evaluate_mas as cw_evaluate_mas

        cw_params = {
            "n_ants": args.n_ants,
            "n_iterations": args.n_iterations,
            "num_agents": args.num_agents,
            "task_name": "CreativeWriting",
            "model": args.model,
            "rho": args.rho,
        }

        if args.cw_train:
            cw_load_batch_fn = lambda: load_cw_train_batch(args.cw_train_batch, args.cw_train_seed)
        else:
            cw_load_batch_fn = lambda: load_cw_batch(args.cw_batch, args.cw_batch_seed)

        best_cw, log_cw, batch_cw = run_aco(
            **cw_params,
            load_batch_fn=cw_load_batch_fn,
            evaluate_fn=cw_evaluate_mas,
            evaluate_kwargs={
                "cw_judge_n_samples": args.cw_judge_n_samples,
                "judge_model": args.cw_judge_model,
            },
        )
        _save_best_config(args.log_dir, "cw", best_cw, log_cw, cw_params)

        # Load a separate eval batch if requested (train mode only), otherwise reuse train batch.
        if args.cw_train and args.cw_eval_batch is not None:
            print(f"Loading separate CW eval batch (n={args.cw_eval_batch}, seed={args.cw_eval_seed})...")
            eval_batch_cw = load_cw_batch(args.cw_eval_batch, args.cw_eval_seed)
        else:
            eval_batch_cw = batch_cw

        report_cw = None
        if eval_batch_cw is not None:
            _, report_cw = cw_evaluate_mas(
                best_cw,
                task_name="CreativeWriting",
                model=args.model,
                judge_model=args.cw_judge_model,
                evaluation_batch=eval_batch_cw,
                cw_judge_n_samples=args.cw_judge_n_samples,
                return_cw_report=True,
            )

        cw_aco_params: dict = {
            **cw_params,
            "judge_n_samples": args.cw_judge_n_samples,
        }
        if args.cw_train:
            cw_aco_params["train_split"] = True
            cw_aco_params["train_batch_size"] = args.cw_train_batch if args.cw_train_batch else 5
            cw_aco_params["train_seed"] = args.cw_train_seed
            cw_aco_params["eval_batch_size"] = args.cw_eval_batch
            cw_aco_params["eval_seed"] = args.cw_eval_seed
        else:
            cw_aco_params["batch_size"] = args.cw_batch
            cw_aco_params["batch_seed"] = args.cw_batch_seed

        out["creative_writing"] = {
            "aco_parameters": cw_aco_params,
            "best_fitness_last_iter": log_cw[-1]["best"] if log_cw else None,
            "best_configuration_str": str(best_cw),
            "reportable": report_cw,
        }

    stem = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tracks = []
    if not args.skip_tp:
        tracks.append("tp")
    if not args.skip_np:
        tracks.append(f"np_{args.np_kind}")
    if not args.skip_cw:
        tracks.append("cw")
    kind_tag = "_".join(tracks) if tracks else "dry"
    log_path = args.log_dir / f"{stem}_{kind_tag}_aco_run.json"
    payload = _json_safe(out)
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {log_path}")

    if not args.no_mirror_folder_logs:
        try:
            rel_main = log_path.relative_to(root)
        except ValueError:
            rel_main = log_path
        ts = out["timestamp_utc"]
        if out.get("travel_planner") is not None:
            tp_mirror = {
                "written_by": "run.py",
                "timestamp_utc": ts,
                "full_run_json": str(rel_main).replace("\\", "/"),
                "travel_planner": out["travel_planner"],
            }
            (root / "aco_final").mkdir(parents=True, exist_ok=True)
            (root / "aco_final" / "run_latest.json").write_text(
                json.dumps(_json_safe(tp_mirror), indent=2), encoding="utf-8"
            )
            print(f"Wrote {root / 'aco_final' / 'run_latest.json'}")
        if out.get("natural_plan") is not None:
            np_mirror = {
                "written_by": "run.py",
                "timestamp_utc": ts,
                "full_run_json": str(rel_main).replace("\\", "/"),
                "natural_plan": out["natural_plan"],
            }
            (root / "aco_np").mkdir(parents=True, exist_ok=True)
            (root / "aco_np" / "run_latest.json").write_text(
                json.dumps(_json_safe(np_mirror), indent=2), encoding="utf-8"
            )
            print(f"Wrote {root / 'aco_np' / 'run_latest.json'}")
        if out.get("creative_writing") is not None:
            cw_mirror = {
                "written_by": "run.py",
                "timestamp_utc": ts,
                "full_run_json": str(rel_main).replace("\\", "/"),
                "creative_writing": out["creative_writing"],
            }
            (root / "aco_cw").mkdir(parents=True, exist_ok=True)
            (root / "aco_cw" / "run_latest.json").write_text(
                json.dumps(_json_safe(cw_mirror), indent=2), encoding="utf-8"
            )
            print(f"Wrote {root / 'aco_cw' / 'run_latest.json'}")


if __name__ == "__main__":
    main()
