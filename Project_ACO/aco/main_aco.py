import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hf_cache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_hf_cache, "datasets"))

from .pheromone import PheromoneMatrix
from .construction import construct_solution


def run_aco(
    n_ants=20,
    n_iterations=10,
    num_agents=5,
    task_name="Task",
    model="meta/llama-3.1-8b-instruct",
    load_batch_fn=None,
    evaluate_fn=None,
    evaluate_kwargs=None,
    rho=0.1,
    Q=1.0,
    tau_init=1.0,
    tau_min=0.01,
    tau_max=10.0,
    elite_only=True,
):
    """
    Dataset-agnostic ACO loop over MAS configurations.

    Args:
        load_batch_fn: Callable ``() -> list[dict]`` that returns the evaluation batch.
            Called once before the loop. If None or raises, evaluation_batch is None.
        evaluate_fn: Callable ``(individual, task_name, model, evaluation_batch, **evaluate_kwargs)
            -> (float,)``. Called once per ant per iteration.
        evaluate_kwargs: Extra keyword args forwarded to evaluate_fn on every call.

    Returns:
        ``(best_solution, logbook, evaluation_batch)``
    """
    if evaluate_kwargs is None:
        evaluate_kwargs = {}

    print(f"--- Starting ACO ({task_name}) ---")
    print(
        f"Task: {task_name} | Model: {model} | Ants: {n_ants} | "
        f"Iterations: {n_iterations} | rho: {rho}"
    )

    evaluation_batch = None
    if load_batch_fn is not None:
        try:
            evaluation_batch = load_batch_fn()
            print(f"Loaded {len(evaluation_batch)} samples.")
        except Exception as e:
            print(f"Warning: could not load dataset: {e}")

    pheromone = PheromoneMatrix(
        num_agents, tau_init=tau_init, tau_min=tau_min, tau_max=tau_max
    )
    best_solution = None
    best_fitness = -float("inf")
    logbook = []

    for iteration in range(n_iterations):
        solutions = []
        fitnesses = []
        print(f"\n{'#'*60}")
        print(f"  ITERATION {iteration + 1}/{n_iterations} | {n_ants} ants")
        print(f"{'#'*60}")

        ant_counter = [0]

        def _evaluate_ant(ant_idx):
            sol = construct_solution(pheromone, num_agents)
            print(f"\n  >>> ANT {ant_idx + 1}/{n_ants} starting | config: {sol}")
            fit, = evaluate_fn(
                sol,
                task_name=task_name,
                model=model,
                evaluation_batch=evaluation_batch,
                **evaluate_kwargs,
            )
            print(f"  <<< ANT {ant_idx + 1}/{n_ants} done | fitness: {fit:.2f}")
            return sol, fit

        with ThreadPoolExecutor(max_workers=n_ants) as pool:
            futures = [pool.submit(_evaluate_ant, i) for i in range(n_ants)]
            for future in as_completed(futures):
                sol, fit = future.result()
                solutions.append(sol)
                fitnesses.append(fit)

        iter_best_idx = max(range(n_ants), key=lambda k: fitnesses[k])
        if fitnesses[iter_best_idx] > best_fitness:
            best_fitness = fitnesses[iter_best_idx]
            best_solution = solutions[iter_best_idx].clone()

        avg_fit = sum(fitnesses) / len(fitnesses)
        min_fit = min(fitnesses)
        max_fit = max(fitnesses)
        print(
            f"  Iter {iteration:3d} | avg={avg_fit:.2f} | min={min_fit:.2f} | "
            f"max={max_fit:.2f} | best_ever={best_fitness:.2f}"
        )
        logbook.append(
            {
                "iter": iteration,
                "avg": avg_fit,
                "min": min_fit,
                "max": max_fit,
                "best": best_fitness,
            }
        )

        pheromone.evaporate(rho)
        if elite_only:
            delta = Q * best_fitness / 100.0
            pheromone.deposit(best_solution, delta)
        else:
            max_f = max(fitnesses) if fitnesses and max(fitnesses) > 0 else 1.0
            for sol, fit in zip(solutions, fitnesses):
                delta = Q * (fit / max_f)
                pheromone.deposit(sol, delta)

    print("\n--- ACO Optimization Complete ---")
    print(f"Best Configuration: {best_solution}")
    print(f"Best Fitness Score: {best_fitness:.2f}")
    return best_solution, logbook, evaluation_batch
