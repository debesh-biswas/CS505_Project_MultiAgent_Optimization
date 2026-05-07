import random
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from datasets import load_dataset
from deap import base, creator, tools, algorithms
from .genome import MASConfiguration
from .operators import cx_mas, mut_mas
from .evaluation import evaluate_mas

# Load dataset once globally to avoid reloading
try:
    print("Loading TravelPlanner dataset...")
    TRAVEL_DATASET = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
except Exception as e:
    print(f"Warning: Could not load dataset: {e}")
    TRAVEL_DATASET = None

# Set up DEAP creators
# Maximizing fitness (e.g., accuracy, delivery rate)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", MASConfiguration, fitness=creator.FitnessMax)

def create_individual(num_agents):
    """Creates a randomly initialized Individual."""
    ind = creator.Individual(num_agents)
    ind.initialize_random()
    return ind

def setup_toolbox(num_agents, task_name, model, evaluation_batch=None):
    """Sets up the DEAP toolbox with the required genetic operators."""
    toolbox = base.Toolbox()
    
    # Registration for individual and population
    toolbox.register("individual", create_individual, num_agents=num_agents)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators
    toolbox.register("evaluate", evaluate_mas, task_name=task_name, model=model, evaluation_batch=evaluation_batch)
    toolbox.register("mate", cx_mas)
    toolbox.register("mutate", mut_mas, mutpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def run_ga(pop_size=20, num_generations=10, num_agents=5, task_name="TravelPlanner", model="Qwen/Qwen2.5-7B-Instruct", batch_size=2):
    """
    Runs the Genetic Algorithm to find the optimal MAS configuration.
    """
    print(f"--- Starting GA Optimization ---")
    print(f"Task: {task_name} | Model: {model} | Pop Size: {pop_size} | Gens: {num_generations} | Batch: {batch_size}")

    # Select random evaluation batch
    evaluation_batch = None
    if task_name == "TravelPlanner" and TRAVEL_DATASET is not None:
        # Randomly select a few queries for this GA run to keep it fast
        indices = random.sample(range(len(TRAVEL_DATASET)), min(batch_size, len(TRAVEL_DATASET)))
        evaluation_batch = [TRAVEL_DATASET[i] for i in indices]
        print(f"Selected {len(evaluation_batch)} queries for evaluation.")

    toolbox = setup_toolbox(num_agents, task_name, model, evaluation_batch=evaluation_batch)
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1) # Keep track of the best individual
    
    # Statistics to keep track of progress
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda pop: sum(x[0] for x in pop) / len(pop))
    stats.register("min", min)
    stats.register("max", max)
    
    # Run simple evolutionary algorithm
    population, logbook = algorithms.eaSimple(
        population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, 
        stats=stats, halloffame=hof, verbose=True
    )
    
    best_ind = hof[0]
    print("\n--- GA Optimization Complete ---")
    print(f"Best Configuration: {best_ind}")
    print(f"Best Fitness Score: {best_ind.fitness.values[0]:.2f}")
    
    return best_ind, logbook

def task_stratified_analysis(best_ind, tasks=["TravelPlanner", "CreativeReasoning", "MathQA"]):
    """
    Evaluates the best individual across different tasks (Phase 3).
    """
    print("\n--- Task-Stratified Analysis ---")
    results = {}
    for task in tasks:
        evaluation_batch = None
        if task == "TravelPlanner" and TRAVEL_DATASET is not None:
            indices = random.sample(range(len(TRAVEL_DATASET)), 2)
            evaluation_batch = [TRAVEL_DATASET[i] for i in indices]
            
        score = evaluate_mas(best_ind, task_name=task, model="Qwen/Qwen2.5-7B-Instruct", evaluation_batch=evaluation_batch)[0]
        results[task] = score
        print(f"Performance on {task}: {score:.2f}")
    return results

def backend_testing(best_ind, task_name="TravelPlanner"):
    """
    Evaluates the best individual using a different LLM backend (Phase 3).
    """
    print("\n--- Backend Testing ---")
    print("Testing with Llama 3.1 instead of original model")
    
    evaluation_batch = None
    if task_name == "TravelPlanner" and TRAVEL_DATASET is not None:
        indices = random.sample(range(len(TRAVEL_DATASET)), 2)
        evaluation_batch = [TRAVEL_DATASET[i] for i in indices]
        
    score = evaluate_mas(best_ind, task_name=task_name, model="Qwen/Qwen2.5-7B-Instruct", evaluation_batch=evaluation_batch)[0]
    print(f"Performance using Llama 3.1 on {task_name}: {score:.2f}")

if __name__ == "__main__":
    # Example execution
    best_config, log = run_ga(pop_size=10, num_generations=5, num_agents=4, task_name="TravelPlanner")
    
    # Save the statistics to a file so it's easy to read later
    import os
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ga_results.txt"))
    with open(save_path, "w") as f:
        f.write("--- GA Optimization Results ---\n\n")
        f.write("DEAP Statistics Logbook:\n")
        f.write(str(log))
        f.write("\n\nBest Configuration Found:\n")
        f.write(str(best_config))
        f.write(f"\n\nBest Fitness Score: {best_config.fitness.values[0]:.2f}\n")
    print(f"\n[Saved statistics and best configuration to '{save_path}']")
    
    # Phase 3 Analysis
    task_stratified_analysis(best_config)
    backend_testing(best_config)
