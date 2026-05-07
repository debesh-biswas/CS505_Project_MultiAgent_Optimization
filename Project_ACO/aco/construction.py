import random
from .genome import MASConfiguration, AVAILABLE_ROLES, AVAILABLE_CAPABILITIES


def _weighted_choice(items, probs):
    """Select one item according to the given probability distribution."""
    r = random.random()
    cumsum = 0.0
    for item, p in zip(items, probs):
        cumsum += p
        if r <= cumsum:
            return item
    return items[-1]  # floating-point safety fallback


def construct_solution(pheromone, num_agents):
    """
    Build one ant's solution by sampling each component from pheromone trails.

    For each agent position: sample role then capability.
    For each directed edge (i,j): activate with probability proportional to tau_on.
    """
    config = MASConfiguration(num_agents)
    config.agents = []

    for i in range(num_agents):
        role = _weighted_choice(AVAILABLE_ROLES, pheromone.role_probs(i))
        cap = _weighted_choice(AVAILABLE_CAPABILITIES, pheromone.cap_probs(i))
        config.agents.append({"role": role, "capability": cap})

    config.links = []
    for i in range(num_agents):
        row = []
        for j in range(num_agents):
            if i == j:
                row.append(0)
            else:
                p_on = pheromone.link_prob_on(i, j)
                row.append(1 if random.random() < p_on else 0)
        config.links.append(row)

    return config
