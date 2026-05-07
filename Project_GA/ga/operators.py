import random
from .genome import AVAILABLE_ROLES, AVAILABLE_CAPABILITIES

def cx_mas(ind1, ind2):
    """
    Crossover operator for MAS configurations.
    Performs uniform crossover on agents and links.
    """
    assert ind1.num_agents == ind2.num_agents, "Individuals must have the same number of agents for crossover."

    for i in range(ind1.num_agents):
        if random.random() < 0.5:
            ind1.agents[i], ind2.agents[i] = ind2.agents[i], ind1.agents[i]

    for i in range(ind1.num_agents):
        for j in range(ind1.num_agents):
            if random.random() < 0.5:
                ind1.links[i][j], ind2.links[i][j] = ind2.links[i][j], ind1.links[i][j]

    return ind1, ind2

def mut_mas(ind, mutpb):
    """
    Mutation operator for MAS configurations.
    mutpb: Probability of mutating each attribute (agent role, capability, or link).
    """
    for i in range(ind.num_agents):
        if random.random() < mutpb:
            if random.random() < 0.5:
                ind.agents[i]["role"] = random.choice(AVAILABLE_ROLES)
        if random.random() < mutpb:
            ind.agents[i]["capability"] = random.choice(AVAILABLE_CAPABILITIES)

    for i in range(ind.num_agents):
        for j in range(ind.num_agents):
            if i != j and random.random() < mutpb:
                ind.links[i][j] = 1 - ind.links[i][j]

    return ind,
