import random

# Example search space for Roles (Ri) and Capabilities (Ci)
AVAILABLE_ROLES = [
    "Planner",
    "Executor",
    "Critic",
    "Researcher",
    "Summarizer",
    "Coder"
]

AVAILABLE_CAPABILITIES = [
    "WebSearch",
    "CodeExecution",
    "FileIO",
    "DataAnalysis",
    "MathSolver",
    "None"
]

class MASConfiguration:
    """
    Represents a candidate Multi-Agent System configuration (an Individual in GA).
    """
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.agents = []
        self.links = [] # Adjacency matrix representation

    def initialize_random(self):
        """Randomly initialize agents and communication links."""
        # Initialize agents with random role and capability
        self.agents = []
        for _ in range(self.num_agents):
            agent = {
                "role": random.choice(AVAILABLE_ROLES),
                "capability": random.choice(AVAILABLE_CAPABILITIES)
            }
            self.agents.append(agent)
            
        # Initialize communication links (Directed graph as adjacency matrix)
        # 1 means agent i can send messages to agent j
        self.links = []
        for i in range(self.num_agents):
            row = []
            for j in range(self.num_agents):
                if i == j:
                    row.append(0) # No self loops
                else:
                    row.append(random.choice([0, 1]))
            self.links.append(row)
            
    def clone(self):
        """Creates a deep copy of this configuration."""
        new_mas = MASConfiguration(self.num_agents)
        new_mas.agents = [dict(a) for a in self.agents]
        new_mas.links = [list(row) for row in self.links]
        return new_mas

    def __str__(self):
        agents_str = ", ".join([f"{a['role']}({a['capability']})" for a in self.agents])
        return f"MAS(Agents: [{agents_str}])"
