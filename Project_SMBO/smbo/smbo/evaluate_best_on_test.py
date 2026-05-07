# #!/usr/bin/env python3
# """Evaluate the best SMBO-found config on a held-out test set.

# Usage:
#     python evaluate_best_on_test.py --smbo_results smbo_out/smbo_results.json --test_indices smbo_out/test_indices.json --test_size 50 --model meta/llama-4-maverick-17b-128e-instruct
# """
# import argparse
# import json
# import os
# import sys
# import random

# # Ensure project root is in path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # If we're in ga_mas subdirectory, go up one level
# if os.path.basename(current_dir) == 'ga_mas':
#     project_root = os.path.dirname(current_dir)
# else:
#     project_root = current_dir

# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from ga_mas.genome import MASConfiguration
# from ga_mas.evaluation import evaluate_mas


# def create_test_indices(test_size: int = 50, seed: int = 42, out_path: str = None):
#     """Create and save fixed test set indices from TravelPlanner validation split."""
#     try:
#         from datasets import load_dataset
#         print(f"Creating test set with {test_size} samples (seed={seed})...")
#         ds = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
        
#         rng = random.Random(seed)
#         indices = rng.sample(range(len(ds)), min(test_size, len(ds)))
        
#         if out_path:
#             os.makedirs(os.path.dirname(out_path), exist_ok=True)
#             with open(out_path, 'w') as f:
#                 json.dump(indices, f, indent=2)
#             print(f"Saved test indices to {out_path}")
        
#         return indices
#     except Exception as e:
#         print(f"Failed to create test indices: {e}")
#         return None


# def load_test_set(indices: list, batch_size: int = None):
#     """Load TravelPlanner samples for given indices."""
#     try:
#         from datasets import load_dataset
#         ds = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
#         if batch_size:
#             indices = indices[:batch_size]
#         batch = [ds[i] for i in indices]
#         print(f"Loaded {len(batch)} test samples.")
#         return batch
#     except Exception as e:
#         print(f"Failed to load test set: {e}")
#         return None


# def load_best_config(smbo_results_path: str):
#     """Load best_config from SMBO results JSON."""
#     try:
#         with open(smbo_results_path, 'r') as f:
#             results = json.load(f)
        
#         best_cfg = results.get('best_config')
#         if not best_cfg:
#             print(f"No 'best_config' found in {smbo_results_path}")
#             return None
        
#         # Reconstruct MASConfiguration from dict
#         num_agents = len(best_cfg.get('agents', []))
#         mas = MASConfiguration(num_agents)
#         mas.agents = best_cfg['agents']
#         mas.links = best_cfg['links']
        
#         print(f"Loaded best config: {num_agents} agents, best_score={results.get('best_score')}")
#         return mas
#     except Exception as e:
#         print(f"Failed to load best config: {e}")
#         return None


# def main():
#     parser = argparse.ArgumentParser(description='Evaluate best SMBO config on held-out test set')
    
#     # Detect if we're being run from ga_mas/ or project root
#     cwd = os.getcwd()
#     if cwd.endswith('ga_mas'):
#         default_out_dir = os.path.join('..', 'smbo_out')
#     else:
#         default_out_dir = 'smbo_out'
    
#     parser.add_argument('--smbo_results', type=str, default=os.path.join(default_out_dir, 'smbo_results.json'),
#                         help='Path to SMBO results JSON')
#     parser.add_argument('--test_indices', type=str, default=os.path.join(default_out_dir, 'test_indices.json'),
#                         help='Path to save/load test set indices')
#     parser.add_argument('--test_size', type=int, default=50,
#                         help='Number of test samples (if creating new indices)')
#     parser.add_argument('--test_seed', type=int, default=42,
#                         help='Seed for deterministic test set creation')
#     parser.add_argument('--model', type=str, default='meta/llama-4-maverick-17b-128e-instruct',
#                         help='LLM model for evaluation')
#     parser.add_argument('--batch_size', type=int, default=None,
#                         help='Limit test batch size (default: use all)')
#     parser.add_argument('--out_dir', type=str, default=default_out_dir,
#                         help='Output directory for test results')
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     # 1. Create or load test indices
#     if os.path.exists(args.test_indices):
#         print(f"Loading existing test indices from {args.test_indices}")
#         with open(args.test_indices, 'r') as f:
#             test_indices = json.load(f)
#     else:
#         test_indices = create_test_indices(
#             test_size=args.test_size,
#             seed=args.test_seed,
#             out_path=args.test_indices
#         )
#         if test_indices is None:
#             return

#     # 2. Load best config
#     best_mas = load_best_config(args.smbo_results)
#     if best_mas is None:
#         return

#     # 3. Load test set
#     test_batch = load_test_set(test_indices, batch_size=args.batch_size)
#     if test_batch is None:
#         return

#     # 4. Evaluate best config on test set
#     print(f"\nEvaluating best config on {len(test_batch)} test samples...")
#     try:
#         test_score = evaluate_mas(
#             individual=best_mas,
#             task_name='TravelPlanner',
#             model=args.model,
#             evaluation_batch=test_batch
#         )[0]
#         print(f"Test score: {test_score:.2f}")
#     except Exception as e:
#         print(f"Evaluation failed: {e}")
#         return

#     # 5. Save results
#     test_results = {
#         'test_size': len(test_batch),
#         'test_score': float(test_score),
#         'model': args.model,
#         'best_config': best_mas.agents,
#         'best_config_links': best_mas.links,
#     }
    
#     out_path = os.path.join(args.out_dir, 'test_evaluation.json')
#     with open(out_path, 'w') as f:
#         json.dump(test_results, f, indent=2)
    
#     print(f"\nTest evaluation results saved to {out_path}")
#     print(json.dumps(test_results, indent=2))


# if __name__ == '__main__':
#     main()



















# below with more constrained score


#!/usr/bin/env python3
"""Evaluate the best SMBO-found config on a held-out test set.

Usage:
    python evaluate_best_on_test.py --smbo_results smbo_out/smbo_results.json --test_indices smbo_out/test_indices.json --test_size 50 --model meta/llama-4-maverick-17b-128e-instruct
"""
import argparse
import json
import os
import sys
import random

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
# If we're in ga_mas subdirectory, go up one level
if os.path.basename(current_dir) == 'ga_mas':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ga_mas.genome import MASConfiguration
from ga_mas.evaluation import evaluate_mas


def create_test_indices(test_size: int = 50, seed: int = 42, out_path: str = None):
    """Create and save fixed test set indices from TravelPlanner validation split."""
    try:
        from datasets import load_dataset
        print(f"Creating test set with {test_size} samples (seed={seed})...")
        ds = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
        
        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(test_size, len(ds)))
        
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(indices, f, indent=2)
            print(f"Saved test indices to {out_path}")
        
        return indices
    except Exception as e:
        print(f"Failed to create test indices: {e}")
        return None


def load_test_set(indices: list, batch_size: int = None):
    """Load TravelPlanner samples for given indices."""
    try:
        from datasets import load_dataset
        ds = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
        if batch_size:
            indices = indices[:batch_size]
        batch = [ds[i] for i in indices]
        print(f"Loaded {len(batch)} test samples.")
        return batch
    except Exception as e:
        print(f"Failed to load test set: {e}")
        return None


def load_best_config(smbo_results_path: str):
    """Load best_config from SMBO results JSON."""
    try:
        with open(smbo_results_path, 'r') as f:
            results = json.load(f)

        best_cfg = results.get('best_config')
        # Backward compatibility: recover best config from history if needed.
        if not best_cfg:
            history = results.get('history', [])
            if history and isinstance(history, list):
                valid = [h for h in history if isinstance(h, dict) and 'config' in h and 'score' in h]
                if valid:
                    best_item = max(valid, key=lambda x: float(x.get('score', -1e9)))
                    best_cfg = best_item.get('config')
                    print("Recovered best config from 'history' section.")

        if not best_cfg:
            print(f"No 'best_config' or recoverable 'history' found in {smbo_results_path}")
            print(f"Available top-level keys: {list(results.keys())}")
            return None
        
        # Reconstruct MASConfiguration from dict
        num_agents = len(best_cfg.get('agents', []))
        mas = MASConfiguration(num_agents)
        mas.agents = best_cfg['agents']
        mas.links = best_cfg['links']
        
        print(f"Loaded best config: {num_agents} agents, best_score={results.get('best_score')}")
        return mas
    except Exception as e:
        print(f"Failed to load best config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate best SMBO config on held-out test set')
    
    # Detect if we're being run from ga_mas/ or project root
    cwd = os.getcwd()
    if cwd.endswith('ga_mas'):
        default_out_dir = os.path.join('..', 'smbo_out')
    else:
        default_out_dir = 'smbo_out'
    
    parser.add_argument('--smbo_results', type=str, default=os.path.join(default_out_dir, 'smbo_results.json'),
                        help='Path to SMBO results JSON')
    parser.add_argument('--test_indices', type=str, default=os.path.join(default_out_dir, 'test_indices.json'),
                        help='Path to save/load test set indices')
    parser.add_argument('--test_size', type=int, default=50,
                        help='Number of test samples (if creating new indices)')
    parser.add_argument('--test_seed', type=int, default=42,
                        help='Seed for deterministic test set creation')
    parser.add_argument('--model', type=str, default='meta/llama-4-maverick-17b-128e-instruct',
                        help='LLM model for evaluation')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Limit test batch size (default: use all)')
    parser.add_argument('--out_dir', type=str, default=default_out_dir,
                        help='Output directory for test results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Create or load test indices
    if os.path.exists(args.test_indices):
        print(f"Loading existing test indices from {args.test_indices}")
        with open(args.test_indices, 'r') as f:
            test_indices = json.load(f)
    else:
        test_indices = create_test_indices(
            test_size=args.test_size,
            seed=args.test_seed,
            out_path=args.test_indices
        )
        if test_indices is None:
            return

    # 2. Load best config
    best_mas = load_best_config(args.smbo_results)
    if best_mas is None:
        return

    # 3. Load test set
    test_batch = load_test_set(test_indices, batch_size=args.batch_size)
    if test_batch is None:
        return

    # 4. Evaluate best config on test set
    print(f"\nEvaluating best config on {len(test_batch)} test samples...")
    try:
        test_score = evaluate_mas(
            individual=best_mas,
            task_name='TravelPlanner',
            model=args.model,
            evaluation_batch=test_batch
        )[0]
        print(f"Test score: {test_score:.2f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return

    # 5. Save results
    test_results = {
        'test_size': len(test_batch),
        'test_score': float(test_score),
        'model': args.model,
        'best_config': best_mas.agents,
        'best_config_links': best_mas.links,
    }
    
    out_path = os.path.join(args.out_dir, 'test_evaluation.json')
    with open(out_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest evaluation results saved to {out_path}")
    print(json.dumps(test_results, indent=2))


if __name__ == '__main__':
    main()

