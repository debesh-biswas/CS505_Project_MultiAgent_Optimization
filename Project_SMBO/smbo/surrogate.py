"""
SMBOOptimizer and GenomeEncoder for surrogate-model-based MAS optimization.

Uses a RandomForest surrogate with UCB acquisition over a sampled candidate pool.
"""
import os
import random
import json
from typing import List, Callable, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .genome import MASConfiguration


class GenomeEncoder:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def encode(self, individual: MASConfiguration) -> np.ndarray:
        # Simple integer encoding: role_idx, cap_idx, then flattened links
        vec = []
        for a in individual.agents:
            # roles and capabilities are stored as strings; map via index in available lists
            try:
                from . import genome as _g
                role_idx = _g.AVAILABLE_ROLES.index(a.get('role'))
                cap_idx = _g.AVAILABLE_CAPABILITIES.index(a.get('capability'))
            except Exception:
                role_idx = 0
                cap_idx = 0
            vec.append(role_idx)
            vec.append(cap_idx)

        # Flatten links row-major
        for row in individual.links:
            for v in row:
                vec.append(int(v))

        return np.array(vec, dtype=float)


class SMBOOptimizer:
    """
    Surrogate-assisted SMBO optimizer for MAS configuration.

    Uses a RandomForest surrogate and UCB acquisition over a sampled pool.
    """
    def __init__(self, num_agents: int, n_init: int = 30, iterations: int = 40,
                 pool_size: int = 500, top_k: int = 5, kappa: float = 1.0,
                 random_seed: int = 0, save_dir: str = None):
        self.num_agents = num_agents
        self.n_init = n_init
        self.iterations = iterations
        self.pool_size = pool_size
        self.top_k = top_k
        self.kappa = kappa
        self.random = random.Random(random_seed)
        self.encoder = GenomeEncoder(num_agents)
        self.save_dir = save_dir or os.path.abspath(os.getcwd())
        os.makedirs(self.save_dir, exist_ok=True)

        # Storage
        self.X = []
        self.y = []
        # Keep the actual evaluated configurations in same order as X/y
        self.configs = []
        self.evaluated_hashes = set()

    def _hash_feat(self, feat: np.ndarray) -> str:
        return feat.tobytes()

    def _random_candidate(self) -> MASConfiguration:
        ind = MASConfiguration(self.num_agents)
        ind.initialize_random()
        return ind

    def _train_surrogate(self) -> RandomForestRegressor:
        if not self.X:
            return None
        X = np.vstack(self.X)
        y = np.array(self.y)
        rf = RandomForestRegressor(n_estimators=100, random_state=0)
        rf.fit(X, y)
        return rf

    def _predict_ensemble(self, model: RandomForestRegressor, X_pool: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Use individual tree predictions to compute mean and std
        all_preds = np.stack([est.predict(X_pool) for est in model.estimators_], axis=1)
        mu = all_preds.mean(axis=1)
        sigma = all_preds.std(axis=1)
        return mu, sigma

    def run(self, evaluate_fn: Callable[[MASConfiguration], float],
            budget: int = 200) -> dict:
        """
        Run the SMBO loop.

        evaluate_fn: function that accepts a MASConfiguration and returns a (scalar,) tuple.
        budget: total number of expensive evaluations allowed (including initial).
        Returns a dictionary with logs and best-found candidates.
        """
        # 1) initial random evaluations
        init_evals = min(self.n_init, budget)
        print(f"SMBO: performing {init_evals} initial random evaluations...")
        for _ in range(init_evals):
            ind = self._random_candidate()
            feat = self.encoder.encode(ind)
            h = self._hash_feat(feat)
            if h in self.evaluated_hashes:
                continue
            try:
                score = float(evaluate_fn(ind)[0])
            except Exception:
                score = 0.0
            self.X.append(feat)
            self.y.append(score)
            try:
                self.configs.append({'agents': ind.agents, 'links': ind.links})
            except Exception:
                self.configs.append({'agents': [], 'links': []})
            self.evaluated_hashes.add(h)

        evaluations = len(self.y)

        # Main loop
        iteration = 0
        while evaluations < budget and iteration < self.iterations:
            iteration += 1
            print(f"SMBO Iteration {iteration}: training surrogate on {len(self.y)} points...")
            rf = self._train_surrogate()
            if rf is None:
                break

            # Sample a candidate pool and score with acquisition
            pool_inds = [self._random_candidate() for _ in range(self.pool_size)]
            X_pool = np.vstack([self.encoder.encode(p) for p in pool_inds])
            mu, sigma = self._predict_ensemble(rf, X_pool)
            ucb = mu + self.kappa * sigma

            # Sort pool by UCB and pick top_k unseen
            order = np.argsort(-ucb)
            selected = []
            for idx in order:
                feat = X_pool[idx]
                h = self._hash_feat(feat)
                if h in self.evaluated_hashes:
                    continue
                selected.append((idx, pool_inds[idx], feat))
                if len(selected) >= self.top_k:
                    break

            if not selected:
                print("No new candidates found in pool; increasing pool size or ending.")
                break

            # Evaluate selected candidates
            for idx, ind, feat in selected:
                try:
                    score = float(evaluate_fn(ind)[0])
                except Exception:
                    score = 0.0
                self.X.append(feat)
                self.y.append(score)
                try:
                    self.configs.append({'agents': ind.agents, 'links': ind.links})
                except Exception:
                    self.configs.append({'agents': [], 'links': []})
                self.evaluated_hashes.add(self._hash_feat(feat))
                evaluations += 1
                print(f" Evaluated candidate (score={score:.2f}) | total evals: {evaluations}/{budget}")
                if evaluations >= budget:
                    break

        # Summarize
        best_idx = int(np.argmax(self.y)) if self.y else None
        best_score = float(self.y[best_idx]) if best_idx is not None else None

        result = {
            'best_score': best_score,
            'evaluations': len(self.y),
            'all_scores': self.y,
            'history': [
                {'config': self.configs[i], 'score': float(self.y[i])}
                for i in range(len(self.y))
            ] if len(self.y) == len(self.configs) else [],
            'best_config': (self.configs[best_idx] if best_idx is not None and best_idx < len(self.configs) else None),
        }

        # Save results
        out_path = os.path.join(self.save_dir, 'smbo_results.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"SMBO complete. Best score: {best_score}. Results saved to {out_path}")
        return result
