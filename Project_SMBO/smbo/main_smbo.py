import os

_hf_cache = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Project", "hf_cache")
)
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_hf_cache, "datasets"))

from .surrogate import SMBOOptimizer
from .genome import MASConfiguration


def run_smbo(
    n_init=30,
    iterations=20,
    pool_size=500,
    top_k=5,
    kappa=1.0,
    num_agents=4,
    task_name="Task",
    model="meta/llama-4-maverick-17b-128e-instruct",
    load_batch_fn=None,
    evaluate_fn=None,
    evaluate_kwargs=None,
    random_seed=0,
    save_dir=None,
):
    """
    Dataset-agnostic SMBO loop over MAS configurations.

    Args:
        load_batch_fn: Callable ``() -> list[dict]`` that returns the evaluation batch.
            Called once before the loop. If None or raises, evaluation_batch is None.
        evaluate_fn: Callable ``(individual, task_name, model, evaluation_batch, **evaluate_kwargs)
            -> (float,)``. Called once per candidate per surrogate iteration.
        evaluate_kwargs: Extra keyword args forwarded to evaluate_fn on every call.

    Returns:
        ``(best_solution, logbook, evaluation_batch)`` — same shape as run_aco().
        logbook entries: ``{iter, avg, min, max, best}`` — one per surrogate iteration.
    """
    if evaluate_kwargs is None:
        evaluate_kwargs = {}

    print(f"--- Starting SMBO ({task_name}) ---")
    print(
        f"Task: {task_name} | Model: {model} | n_init: {n_init} | "
        f"Iterations: {iterations} | top_k: {top_k} | kappa: {kappa}"
    )

    evaluation_batch = None
    if load_batch_fn is not None:
        try:
            evaluation_batch = load_batch_fn()
            print(f"Loaded {len(evaluation_batch)} samples.")
        except Exception as e:
            print(f"Warning: could not load dataset: {e}")

    def _inner_eval(individual):
        fit, = evaluate_fn(
            individual,
            task_name=task_name,
            model=model,
            evaluation_batch=evaluation_batch,
            **evaluate_kwargs,
        )
        return float(fit)

    budget = n_init + iterations * top_k

    optimizer = SMBOOptimizer(
        num_agents=num_agents,
        n_init=n_init,
        iterations=iterations,
        pool_size=pool_size,
        top_k=top_k,
        kappa=kappa,
        random_seed=random_seed,
        save_dir=save_dir,
    )

    result = optimizer.run(_inner_eval, budget)

    # Reconstruct best_solution as MASConfiguration from the saved dict.
    best_solution = MASConfiguration(num_agents)
    best_cfg = result["best_config"]
    best_solution.agents = best_cfg["agents"]
    best_solution.links = best_cfg["links"]

    # Build logbook in ACO-compatible format: one entry per surrogate iteration.
    history = result.get("history", [])
    logbook = []

    # Derive running best from init phase before building iteration entries.
    running_best = -float("inf")
    init_scores = [entry["score"] for entry in history[:n_init]]
    if init_scores:
        running_best = max(init_scores)

    for i in range(iterations):
        start = n_init + i * top_k
        end = n_init + (i + 1) * top_k
        chunk = history[start:end]
        if not chunk:
            break
        scores = [entry["score"] for entry in chunk]
        chunk_max = max(scores)
        if chunk_max > running_best:
            running_best = chunk_max
        logbook.append({
            "iter": i,
            "avg": sum(scores) / len(scores),
            "min": min(scores),
            "max": chunk_max,
            "best": running_best,
        })

    print("\n--- SMBO Optimization Complete ---")
    print(f"Best Configuration: {best_solution}")
    print(f"Best Fitness Score: {result['best_score']:.2f}")
    return best_solution, logbook, evaluation_batch
