import os

_hf_cache = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Project", "hf_cache")
)
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_hf_cache, "datasets"))

import copy
import random
from deap import base, creator, algorithms
from .genome import MASConfiguration
from .operators import cx_mas, mut_mas


def run_ga(
    pop_size=20,
    num_generations=10,
    cxpb=0.5,
    mutpb=0.2,
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
    Dataset-agnostic GA loop over MAS configurations using DEAP.

    Returns (best_solution, logbook, evaluation_batch) — same shape as run_aco() / run_smbo().
    logbook entries: {iter, avg, min, max, best} — one per generation.
    """
    if evaluate_kwargs is None:
        evaluate_kwargs = {}

    print(f"--- Starting GA ({task_name}) ---")
    print(
        f"Task: {task_name} | Model: {model} | pop_size: {pop_size} | "
        f"generations: {num_generations} | cxpb: {cxpb} | mutpb: {mutpb}"
    )

    random.seed(random_seed)

    evaluation_batch = None
    if load_batch_fn is not None:
        try:
            evaluation_batch = load_batch_fn()
            print(f"Loaded {len(evaluation_batch)} samples.")
        except Exception as e:
            print(f"Warning: could not load dataset: {e}")

    def _eval(individual):
        result = evaluate_fn(
            individual,
            task_name=task_name,
            model=model,
            evaluation_batch=evaluation_batch,
            **evaluate_kwargs,
        )
        return (float(result[0]),)

    # Guard against re-registration in repeated calls within the same process.
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", MASConfiguration, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def _make_individual():
        ind = creator.Individual(num_agents)
        ind.initialize_random()
        return ind

    toolbox.register("individual", _make_individual)
    toolbox.register("population", lambda n: [toolbox.individual() for _ in range(n)])
    toolbox.register("evaluate", _eval)
    toolbox.register("mate", cx_mas)
    toolbox.register("mutate", mut_mas, mutpb=mutpb)
    toolbox.register("select", __import__("deap.tools", fromlist=["selTournament"]).selTournament, tournsize=3)
    toolbox.register("clone", copy.deepcopy)

    population = toolbox.population(n=pop_size)

    # Evaluate initial population
    print(f"\n[Gen 0] Evaluating initial population ({pop_size} individuals)...")
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    running_best = max(ind.fitness.values[0] for ind in population)
    logbook = []

    for gen in range(1, num_generations + 1):
        print(f"\n[Gen {gen}/{num_generations}] Selecting and evolving...")

        # Select and deep-copy offspring (preserves DEAP fitness attribute)
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation (invalidates fitness of modified individuals)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Evaluate invalid individuals (those whose fitness was invalidated by operators)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        print(f"  Evaluating {len(invalid)} new/mutated individuals...")
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # Replace population
        population[:] = offspring

        scores = [ind.fitness.values[0] for ind in population]
        gen_max = max(scores)
        if gen_max > running_best:
            running_best = gen_max

        logbook.append({
            "iter": gen,
            "avg": sum(scores) / len(scores),
            "min": min(scores),
            "max": gen_max,
            "best": running_best,
        })
        print(
            f"  Gen {gen} | avg={logbook[-1]['avg']:.2f} | "
            f"min={logbook[-1]['min']:.2f} | max={gen_max:.2f} | best_ever={running_best:.2f}"
        )

    # Find best individual in final population
    best_ind = max(population, key=lambda ind: ind.fitness.values[0])

    # Reconstruct as plain MASConfiguration (strip DEAP fitness)
    best_solution = MASConfiguration(num_agents)
    best_solution.agents = [dict(a) for a in best_ind.agents]
    best_solution.links = [list(row) for row in best_ind.links]

    print("\n--- GA Optimization Complete ---")
    print(f"Best Configuration: {best_solution}")
    print(f"Best Fitness Score: {running_best:.2f}")
    return best_solution, logbook, evaluation_batch
