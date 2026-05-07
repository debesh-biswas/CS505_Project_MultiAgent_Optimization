import random

# Core definitions for the Multi-Agent System (MAS)
AVAILABLE_ROLES = {
    "Planner": "Strategic Architect. You lead the synthesis of findings. CRITICAL: You are strictly forbidden from using placeholders like '<Hotel Name>' or empty quotes. You MUST fill every JSON field with real data from the team.",
    "Executor": "Technical Specialist. You execute tool requests. Ensure only one tool is active per turn.",
    "Critic": "Quality Assurance. Rigorously audit plans. Ensure the team follows the 'One Tool Call per Turn' protocol.",
    "Researcher": "Information Specialist. Use tools for ground-truth data. CRITICAL: You MUST only call ONE tool per message (e.g., search_flights, THEN wait, THEN search_hotels).",
    "Summarizer": "Communications Lead. Distill research insights. Help the team maintain a step-by-step tool-calling workflow.",
    "Coder": "Computational Expert. Solve problems with code. Only execute one code block or tool at a time."
}

AVAILABLE_CAPABILITIES = {
    "WebSearch": "You have access to live search tools for flights, hotels, restaurants, and attractions. Use them to find real-world data.",
    "CodeExecution": "You can write and execute Python code blocks to solve mathematical problems or process complex data.",
    "FileIO": "You are authorized to read from and write to the local filesystem for persistent storage of trip plans.",
    "DataAnalysis": "You specialize in interpreting large sets of search results and identifying the best options based on multi-variable constraints.",
    "MathSolver": "You have advanced reasoning capabilities for complex budget calculations and time-zone mathematics.",
    "None": "You rely entirely on your internal knowledge and logic to contribute to the team conversation."
}

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
        # Initialize agents with random role and capability keys
        self.agents = []
        for _ in range(self.num_agents):
            agent = {
                "role": random.choice(list(AVAILABLE_ROLES.keys())),
                "capability": random.choice(list(AVAILABLE_CAPABILITIES.keys()))
            }
            self.agents.append(agent)
            
        # Random communication links (1 = exists, 0 = no link)
        self.links = []
        for i in range(self.num_agents):
            row = []
            for j in range(self.num_agents):
                # Ensure it's more likely to have some links than none
                row.append(1 if random.random() > 0.4 else 0)
            self.links.append(row)

    def mutate(self, role_prob=0.2, cap_prob=0.2, link_prob=0.1):
        """Apply random mutations to the MAS configuration."""
        # Mutate agent properties
        for agent in self.agents:
            if random.random() < role_prob:
                agent["role"] = random.choice(list(AVAILABLE_ROLES.keys()))
            if random.random() < cap_prob:
                agent["capability"] = random.choice(list(AVAILABLE_CAPABILITIES.keys()))
                
        # Mutate communication links
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if random.random() < link_prob:
                    self.links[i][j] = 1 - self.links[i][j] # Flip bit

    def __str__(self):
        agent_strs = [f"{a['role']}({a['capability']})" for a in self.agents]
        return f"MAS(Agents: [{', '.join(agent_strs)}])"
