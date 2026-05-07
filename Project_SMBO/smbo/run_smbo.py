# #!/usr/bin/env python3
# """Run SMBO optimizer for MAS configurations.

# Usage: python run_smbo.py --num_agents 4 --budget 200
# """
# import argparse
# import os
# import sys

# from datasets import load_dataset

# # Ensure package importability
# ROOT = os.path.dirname(os.path.abspath(__file__))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# from ga_mas.smbo import SMBOOptimizer
# from ga_mas.evaluation import evaluate_mas


# def load_travelplanner_batch(batch_size: int):
#     try:
#         dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
#         batch_size = min(batch_size, len(dataset))
#         if batch_size <= 0:
#             return None
#         indices = list(range(batch_size))
#         return [dataset[i] for i in indices]
#     except Exception as exc:
#         print(f'Warning: could not load TravelPlanner dataset: {exc}')
#         return None


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_agents', type=int, default=4)
#     parser.add_argument('--budget', type=int, default=200)
#     parser.add_argument('--n_init', type=int, default=30)
#     parser.add_argument('--iterations', type=int, default=40)
#     parser.add_argument('--pool_size', type=int, default=500)
#     parser.add_argument('--top_k', type=int, default=5)
#     parser.add_argument('--kappa', type=float, default=1.0)
#     parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
#     parser.add_argument('--task', type=str, default='TravelPlanner')
#     parser.add_argument('--out_dir', type=str, default='smbo_out')
#     parser.add_argument('--eval_batch_size', type=int, default=2)
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     optimizer = SMBOOptimizer(
#         num_agents=args.num_agents,
#         n_init=args.n_init,
#         iterations=args.iterations,
#         pool_size=args.pool_size,
#         top_k=args.top_k,
#         kappa=args.kappa,
#         save_dir=args.out_dir
#     )

#     evaluation_batch = None
#     if args.task == 'TravelPlanner':
#         evaluation_batch = load_travelplanner_batch(args.eval_batch_size)

#     # Wrap evaluation fn to pass model and task
#     def eval_fn(ind):
#         return evaluate_mas(individual=ind, task_name=args.task, model=args.model, evaluation_batch=evaluation_batch)

#     result = optimizer.run(evaluate_fn=eval_fn, budget=args.budget)
#     print('Done. Summary:', result)


# if __name__ == '__main__':
#     main()

















# #!/usr/bin/env python3
# """Run SMBO optimizer for MAS configurations on TravelPlanner.

# Usage:
#     python run_smbo.py --num_agents 4 --budget 200 --eval_batch_size 5

# For real TravelPlanner evaluation, set NVIDIA_API_KEY environment variable:
#     export NVIDIA_API_KEY='nvapi-...'
#     python run_smbo.py ...
# """
# import argparse
# import os
# import sys

# # Ensure package importability
# ROOT = os.path.dirname(os.path.abspath(__file__))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# from ga_mas.smbo import SMBOOptimizer
# from ga_mas.evaluation import evaluate_mas


# def load_travelplanner_batch(batch_size: int):
#     """Load a batch of TravelPlanner validation samples.
    
#     Args:
#         batch_size: Number of samples to load.
        
#     Returns:
#         List of TravelPlanner samples or None if load fails.
#     """
#     try:
#         from datasets import load_dataset
#         print(f"Loading TravelPlanner validation split ({batch_size} samples)...")
#         dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
#         batch_size = min(batch_size, len(dataset))
#         if batch_size <= 0:
#             return None
#         indices = list(range(batch_size))
#         batch = [dataset[i] for i in indices]
#         print(f"Loaded {len(batch)} TravelPlanner samples.")
#         return batch
#     except Exception as e:
#         print(f"Failed to load TravelPlanner: {e}")
#         return None


# def main():
#     parser = argparse.ArgumentParser(description='SMBO optimization over TravelPlanner')
#     parser.add_argument('--num_agents', type=int, default=4, help='Number of agents in MAS')
#     parser.add_argument('--budget', type=int, default=200, help='Total evaluation budget')
#     parser.add_argument('--n_init', type=int, default=30, help='Initial random evaluations')
#     parser.add_argument('--iterations', type=int, default=40, help='SMBO iterations')
#     parser.add_argument('--pool_size', type=int, default=500, help='Candidate pool size')
#     parser.add_argument('--top_k', type=int, default=5, help='Top candidates to evaluate per iteration')
#     parser.add_argument('--kappa', type=float, default=1.0, help='UCB exploration parameter')
#     parser.add_argument('--eval_batch_size', type=int, default=5, help='TravelPlanner samples per evaluation')
#     parser.add_argument('--model', type=str, default='qwen', help='LLM model for evaluation')
#     parser.add_argument('--out_dir', type=str, default='smbo_out', help='Output directory for results')
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     # Load TravelPlanner evaluation batch
#     evaluation_batch = load_travelplanner_batch(args.eval_batch_size)
#     if evaluation_batch is None:
#         print("Warning: Could not load TravelPlanner dataset. Will use heuristic evaluation.")

#     optimizer = SMBOOptimizer(
#         num_agents=args.num_agents,
#         n_init=args.n_init,
#         iterations=args.iterations,
#         pool_size=args.pool_size,
#         top_k=args.top_k,
#         kappa=args.kappa,
#         save_dir=args.out_dir
#     )

#     # Create evaluation function
#     def eval_fn(ind):
#         return evaluate_mas(
#             individual=ind,
#             task_name='TravelPlanner',
#             model=args.model,
#             evaluation_batch=evaluation_batch
#         )

#     print(f"Starting SMBO optimization: {args.num_agents} agents, budget {args.budget}")
#     result = optimizer.run(evaluate_fn=eval_fn, budget=args.budget)
#     print('Done. Summary:', result)


# if __name__ == '__main__':
#     main()




















# below with train test - TravelPlanner



#!/usr/bin/env python3
# """Run SMBO optimizer for MAS configurations on TravelPlanner.

# Usage:
#     python run_smbo.py --num_agents 4 --budget 200 --eval_batch_size 5

# For real TravelPlanner evaluation, set NVIDIA_API_KEY environment variable:
#     export NVIDIA_API_KEY='nvapi-...'
#     python run_smbo.py ...
# """
# import argparse
# import json
# import os
# import random
# import time
# import sys

# # Ensure package importability
# ROOT = os.path.dirname(os.path.abspath(__file__))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# from ga_mas.smbo import SMBOOptimizer
# from ga_mas.evaluation import evaluate_mas


# def load_travelplanner_batch(batch_size: int, seed: int = 0, indices_path: str = None):
#     """Load a deterministic batch of TravelPlanner validation samples.
    
#     Args:
#         batch_size: Number of samples to load.
#         seed: Seed used when creating a new saved index file.
#         indices_path: Optional path to a JSON file containing fixed dataset indices.
        
#     Returns:
#         List of TravelPlanner samples or None if load fails.
#     """
#     try:
#         from datasets import load_dataset
#         print(f"Loading TravelPlanner validation split ({batch_size} samples)...")
#         dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
#         batch_size = min(batch_size, len(dataset))
#         if batch_size <= 0:
#             return None
#         if indices_path and os.path.exists(indices_path):
#             with open(indices_path, 'r') as f:
#                 indices = json.load(f)
#             indices = [int(i) for i in indices][:batch_size]
#         else:
#             rng = random.Random(seed)
#             indices = rng.sample(range(len(dataset)), batch_size)
#             if indices_path:
#                 os.makedirs(os.path.dirname(indices_path), exist_ok=True)
#                 with open(indices_path, 'w') as f:
#                     json.dump(indices, f, indent=2)
#         batch = [dataset[i] for i in indices]
#         print(f"Loaded {len(batch)} TravelPlanner samples.")
#         return batch
#     except Exception as e:
#         print(f"Failed to load TravelPlanner: {e}")
#         return None


# def main():
#     parser = argparse.ArgumentParser(description='SMBO optimization over TravelPlanner')
#     parser.add_argument('--num_agents', type=int, default=4, help='Number of agents in MAS')
#     parser.add_argument('--budget', type=int, default=200, help='Total evaluation budget')
#     parser.add_argument('--n_init', type=int, default=30, help='Initial random evaluations')
#     parser.add_argument('--iterations', type=int, default=40, help='SMBO iterations')
#     parser.add_argument('--pool_size', type=int, default=500, help='Candidate pool size')
#     parser.add_argument('--top_k', type=int, default=5, help='Top candidates to evaluate per iteration')
#     parser.add_argument('--kappa', type=float, default=1.0, help='UCB exploration parameter')
#     parser.add_argument('--eval_batch_size', type=int, default=5, help='TravelPlanner samples per evaluation')
#     parser.add_argument('--eval_seed', type=int, default=0, help='Seed for deterministic TravelPlanner sample selection')
#     parser.add_argument('--eval_indices_file', type=str, default='', help='Optional JSON file path for fixed TravelPlanner indices')
#     parser.add_argument('--model', type=str, default='qwen', help='LLM model for evaluation')
#     parser.add_argument('--out_dir', type=str, default='smbo_out', help='Output directory for results')
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     # Load TravelPlanner evaluation batch
#     indices_file = args.eval_indices_file or os.path.join(args.out_dir, 'travelplanner_eval_indices.json')
#     evaluation_batch = load_travelplanner_batch(
#         args.eval_batch_size,
#         seed=args.eval_seed,
#         indices_path=indices_file,
#     )
#     if evaluation_batch is None:
#         print("Warning: Could not load TravelPlanner dataset. Will use heuristic evaluation.")

#     run_started_at = time.perf_counter()

#     optimizer = SMBOOptimizer(
#         num_agents=args.num_agents,
#         n_init=args.n_init,
#         iterations=args.iterations,
#         pool_size=args.pool_size,
#         top_k=args.top_k,
#         kappa=args.kappa,
#         save_dir=args.out_dir
#     )

#     # Create evaluation function
#     def eval_fn(ind):
#         return evaluate_mas(
#             individual=ind,
#             task_name='TravelPlanner',
#             model=args.model,
#             evaluation_batch=evaluation_batch
#         )

#     print(f"Starting SMBO optimization: {args.num_agents} agents, budget {args.budget}")
#     result = optimizer.run(evaluate_fn=eval_fn, budget=args.budget)
#     run_elapsed = time.perf_counter() - run_started_at
#     print(f"SMBO runtime: {run_elapsed:.2f} seconds")
#     print('Done. Summary:', result)


# if __name__ == '__main__':
#     main()





























#!/usr/bin/env python3
"""Run SMBO optimizer for MAS configurations on TravelPlanner.

Usage:
    python run_smbo.py --num_agents 4 --budget 200 --eval_batch_size 5

For real TravelPlanner evaluation, set NVIDIA_API_KEY environment variable:
    export NVIDIA_API_KEY='nvapi-...'
    python run_smbo.py ...
"""
import argparse
import json
import importlib
import os
import random
import time
import sys

# Ensure package importability
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ga_mas.smbo import SMBOOptimizer
from ga_mas.evaluation import evaluate_mas


def _ensure_project_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, '..', 'Project')),
        os.path.abspath(os.path.join(here, '..', '..', 'Project')),
        os.path.abspath(os.path.join(here, '..', '..', '..', 'Project')),
    ]
    for cand in candidates:
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
            break


def load_travelplanner_batch(batch_size: int, seed: int = 0, indices_path: str = None):
    """Load a deterministic batch of TravelPlanner validation samples.
    
    Args:
        batch_size: Number of samples to load.
        seed: Seed used when creating a new saved index file.
        indices_path: Optional path to a JSON file containing fixed dataset indices.
        
    Returns:
        List of TravelPlanner samples or None if load fails.
    """
    try:
        from datasets import load_dataset
        print(f"Loading TravelPlanner validation split ({batch_size} samples)...")
        dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
        batch_size = min(batch_size, len(dataset))
        if batch_size <= 0:
            return None
        if indices_path and os.path.exists(indices_path):
            with open(indices_path, 'r') as f:
                indices = json.load(f)
            indices = [int(i) for i in indices][:batch_size]
        else:
            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), batch_size)
            if indices_path:
                os.makedirs(os.path.dirname(indices_path), exist_ok=True)
                with open(indices_path, 'w') as f:
                    json.dump(indices, f, indent=2)
        batch = [dataset[i] for i in indices]
        print(f"Loaded {len(batch)} TravelPlanner samples.")
        return batch
    except Exception as e:
        print(f"Failed to load TravelPlanner: {e}")
        return None


def load_natural_plan_batch(kind: str, batch_size: int = 0, seed: int = 0):
    """Load a deterministic Natural Plan batch."""
    try:
        _ensure_project_path()
        load_np_batch = importlib.import_module("aco.datasets.np.loader").load_np_batch

        print(f"Loading Natural Plan ({kind}) test split ({batch_size or 'all'} samples)...")
        batch = load_np_batch(kind, max_samples=batch_size, seed=seed)
        print(f"Loaded {len(batch)} Natural Plan samples.")
        return batch
    except Exception as e:
        print(f"Failed to load Natural Plan: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='SMBO optimization over TravelPlanner')
    parser.add_argument('--task', choices=['TravelPlanner', 'NaturalPlan'], default='TravelPlanner', help='Dataset/task to optimize')
    parser.add_argument('--num_agents', type=int, default=4, help='Number of agents in MAS')
    parser.add_argument('--budget', type=int, default=200, help='Total evaluation budget')
    parser.add_argument('--n_init', type=int, default=30, help='Initial random evaluations')
    parser.add_argument('--iterations', type=int, default=40, help='SMBO iterations')
    parser.add_argument('--pool_size', type=int, default=500, help='Candidate pool size')
    parser.add_argument('--top_k', type=int, default=5, help='Top candidates to evaluate per iteration')
    parser.add_argument('--kappa', type=float, default=1.0, help='UCB exploration parameter')
    parser.add_argument('--eval_batch_size', type=int, default=5, help='TravelPlanner samples per evaluation')
    parser.add_argument('--eval_seed', type=int, default=0, help='Seed for deterministic TravelPlanner sample selection')
    parser.add_argument('--eval_indices_file', type=str, default='', help='Optional JSON file path for fixed TravelPlanner indices')
    parser.add_argument('--np_kind', type=str, default='trip', choices=['trip', 'meeting', 'calendar'], help='Natural Plan subtype')
    parser.add_argument('--model', type=str, default='qwen', help='LLM model for evaluation')
    parser.add_argument('--out_dir', type=str, default='smbo_out', help='Output directory for results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.task == 'TravelPlanner':
        indices_file = args.eval_indices_file or os.path.join(args.out_dir, 'travelplanner_eval_indices.json')
        evaluation_batch = load_travelplanner_batch(
            args.eval_batch_size,
            seed=args.eval_seed,
            indices_path=indices_file,
        )
        if evaluation_batch is None:
            print("Warning: Could not load TravelPlanner dataset. Will use heuristic evaluation.")
    else:
        evaluation_batch = load_natural_plan_batch(
            args.np_kind,
            batch_size=args.eval_batch_size,
            seed=args.eval_seed,
        )
        if evaluation_batch is None:
            print("Warning: Could not load Natural Plan dataset.")

    run_started_at = time.perf_counter()

    optimizer = SMBOOptimizer(
        num_agents=args.num_agents,
        n_init=args.n_init,
        iterations=args.iterations,
        pool_size=args.pool_size,
        top_k=args.top_k,
        kappa=args.kappa,
        save_dir=args.out_dir
    )

    # Create evaluation function
    def eval_fn(ind):
        return evaluate_mas(
            individual=ind,
            task_name=args.task,
            model=args.model,
            evaluation_batch=evaluation_batch,
            np_kind=args.np_kind,
        )

    print(f"Starting SMBO optimization: {args.num_agents} agents, budget {args.budget}")
    result = optimizer.run(evaluate_fn=eval_fn, budget=args.budget)
    run_elapsed = time.perf_counter() - run_started_at
    print(f"SMBO runtime: {run_elapsed:.2f} seconds")
    print('Done. Summary:', result)


if __name__ == '__main__':
    main()
