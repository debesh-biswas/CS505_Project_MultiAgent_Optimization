import random
import os
import sys
import json
from typing import List, Any, Optional

# Disable HF progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from deap import base, creator, tools, algorithms
from .genome import MASConfiguration
from .operators import cx_mas, mut_mas
from .evaluation import evaluate_mas
from .datasets.tp.loader import load_batch as load_tp_batch
from .datasets.np.loader import load_np_batch

# Set up DEAP creators
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", MASConfiguration, fitness=creator.FitnessMax)

def create_individual(num_agents):
    """Creates a randomly initialized Individual."""
    ind = creator.Individual(num_agents)
    ind.initialize_random()
    return ind

def setup_toolbox(num_agents, task_name, model, evaluation_batch=None, **kwargs):
    """Sets up the DEAP toolbox with the required genetic operators."""
    toolbox = base.Toolbox()
    
    # Registration for individual and population
    toolbox.register("individual", create_individual, num_agents=num_agents)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators
    toolbox.register("evaluate", evaluate_mas, task_name=task_name, model=model, evaluation_batch=evaluation_batch, **kwargs)
    toolbox.register("mate", cx_mas)
    toolbox.register("mutate", mut_mas, mutpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def run_ga(
    pop_size=20, 
    num_generations=10, 
    num_agents=4, 
    task_name="TravelPlanner", 
    model="meta/llama-4-maverick-17b-128e-instruct", 
    batch_size=9,
    batch_seed=42,
    train_split="train",
    **kwargs
):
    """
    Runs the Genetic Algorithm to find the optimal MAS configuration.
    """
    print(f"--- Starting GA Optimization ({task_name}) ---")
    print(f"Model: {model} | Pop Size: {pop_size} | Gens: {num_generations} | Train Batch: {batch_size} ({train_split})")

    # 1. Load the Optimization Batch (The "Training" Set)
    optimization_batch = None
    if task_name == "TravelPlanner":
        optimization_batch = load_tp_batch(tp_split=train_split, batch_size=batch_size, batch_seed=batch_seed)
    elif task_name == "NaturalPlan":
        np_kind = kwargs.get("np_kind", "trip")
        optimization_batch = load_np_batch(kind=np_kind, max_samples=batch_size, seed=batch_seed)
        kwargs["np_kind"] = np_kind

    toolbox = setup_toolbox(num_agents, task_name, model, evaluation_batch=optimization_batch, **kwargs)
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1) 
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda pop: sum(x[0] for x in pop) / len(pop))
    stats.register("min", min)
    stats.register("max", max)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    for gen in range(1, num_generations + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.2)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        hof.update(offspring)
        
        # Replace the current population by the offspring
        population[:] = offspring
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)
        
        # LOG BEST INDIVIDUAL OF THIS GENERATION TO JSON
        best_this_gen = tools.selBest(population, 1)[0]
        gen_data = {
            "generation": gen,
            "best_fitness": best_this_gen.fitness.values[0],
            "agents": best_this_gen.agents,
            "links": best_this_gen.links
        }
        
        # Load existing history or start new list
        history_file = "ga_evolution_history.json"
        history = []
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                try:
                    history = json.load(f)
                except:
                    history = []
        
        history.append(gen_data)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)
            
        print(f"\n>>> [Generation {gen}] Best configuration saved to {history_file}")
    
    best_ind = hof[0]
    print("\n--- GA Optimization Complete ---")
    print(f"Best Configuration: {best_ind}")
    print(f"Best Fitness Score (on training set): {best_ind.fitness.values[0]:.2f}")
    
    # 2. Load the Final Evaluation Batch (e.g., the 180 validation samples)
    print("\n--- Running Final Evaluation on Validation Set ---")
    eval_batch_size = kwargs.get("eval_batch_size", 180)
    eval_split = "validation" if task_name == "TravelPlanner" else "test"
    
    if task_name == "TravelPlanner":
        final_eval_batch = load_tp_batch(tp_split=eval_split, batch_size=eval_batch_size, batch_seed=batch_seed)
    else:
        final_eval_batch = optimization_batch # Fallback for NP

    report = None
    if task_name == "TravelPlanner":
        _, report = evaluate_mas(best_ind, task_name, model, final_eval_batch, return_tp_report=True)
    elif task_name == "NaturalPlan":
        _, report = evaluate_mas(best_ind, task_name, model, final_eval_batch, return_np_report=True, **kwargs)

    return best_ind, logbook, report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GA-MAS optimization.")
    parser.add_argument("--task", choices=["TravelPlanner", "NaturalPlan"], default="TravelPlanner")
    parser.add_argument("--model", default="meta/llama-4-maverick-17b-128e-instruct")
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--gens", type=int, default=5)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--batch", type=int, default=9, help="Number of training samples (SwarmAgentic used 9)")
    parser.add_argument("--split", default="train", help="Split to use for training (SwarmAgentic used 'train')")
    parser.add_argument("--eval-batch", type=int, default=180, help="Number of samples for final evaluation (Official is 180)")
    parser.add_argument("--np-kind", choices=["trip", "meeting", "calendar"], default="trip")
    
    args = parser.parse_args()
    
    best_config, log, report = run_ga(
        pop_size=args.pop_size, 
        num_generations=args.gens, 
        num_agents=args.agents, 
        task_name=args.task,
        model=args.model,
        batch_size=args.batch,
        train_split=args.split,
        eval_batch_size=args.eval_batch,
        np_kind=args.np_kind
    )
    
    # Save the statistics to a file
    import json
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ga_results.json"))
    results = {
        "task": args.task,
        "model": args.model,
        "best_config": str(best_config),
        "best_fitness": best_config.fitness.values[0],
        "report": report,
        "logbook": [dict(it) for it in log]
    }
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved results to '{save_path}']")
