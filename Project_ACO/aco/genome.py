import random

AVAILABLE_ROLES = [
    "Planner",
    "Executor",
    "Critic",
    "Researcher",
    "Summarizer",
    "Coder"
]

ROLE_DESCRIPTIONS = {
    "Planner": (
        "Strategic Architect. Synthesize the team's findings into a structured output plan. "
        "Use ONLY values that teammates have explicitly reported — copy them character-for-character. "
        "Never invent, paraphrase, or infer values not provided by the team."
    ),
    "Executor": "Technical Specialist. Direct execution of tools and data processing.",
    "Critic": (
        "Quality Assurance. Audit the team's output. "
        "Verify that every value in the final answer matches the source data exactly — "
        "reject any paraphrased, shortened, or invented entries. Check all stated constraints."
    ),
    "Researcher": (
        "Information Specialist. Use available tools to gather data. "
        "When reporting results, copy values character-for-character from tool output — "
        "do NOT paraphrase, summarize, or describe. Preserve exact identifiers, names, and numbers."
    ),
    "Summarizer": (
        "Communications Lead. Condense research into clear summaries while preserving "
        "exact values, identifiers, and names from the source data."
    ),
    "Coder": "Computational Expert. Math and code specialist for quantitative reasoning and calculations.",
}

AVAILABLE_CAPABILITIES = [
    "WebSearch",
    "CodeExecution",
    "FileIO",
    "DataAnalysis",
    "MathSolver",
    "None"
]

CAPABILITY_DESCRIPTIONS = {
    "WebSearch": (
        "You have access to live external search and lookup tools. "
        "Use them to retrieve real-world data — do not guess or fabricate values."
    ),
    "CodeExecution": (
        "You can write and execute Python code blocks to solve mathematical problems "
        "or process complex data."
    ),
    "FileIO": (
        "You are authorized to read from and write to the local filesystem for "
        "persistent storage of data and results."
    ),
    "DataAnalysis": (
        "You specialize in interpreting structured data and search results, identifying "
        "the best options under multi-variable constraints."
    ),
    "MathSolver": (
        "You have advanced reasoning capabilities for complex quantitative problems: "
        "arithmetic, scheduling, unit conversion, and multi-step calculations."
    ),
    "None": (
        "You rely entirely on your internal knowledge and logic to contribute "
        "to the team conversation."
    ),
}

class MASConfiguration:
    """
    Represents a candidate Multi-Agent System configuration.
    """
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.agents = []
        self.links = []  # N×N adjacency matrix

    def initialize_random(self):
        self.agents = []
        for _ in range(self.num_agents):
            self.agents.append({
                "role": random.choice(AVAILABLE_ROLES),
                "capability": random.choice(AVAILABLE_CAPABILITIES)
            })

        self.links = []
        for i in range(self.num_agents):
            row = []
            for j in range(self.num_agents):
                if i == j:
                    row.append(0)
                else:
                    row.append(random.choice([0, 1]))
            self.links.append(row)

    def clone(self):
        new_mas = MASConfiguration(self.num_agents)
        new_mas.agents = [dict(a) for a in self.agents]
        new_mas.links = [list(row) for row in self.links]
        return new_mas

    def __str__(self):
        agents_str = ", ".join([f"{a['role']}({a['capability']})" for a in self.agents])
        return f"MAS(Agents: [{agents_str}])"
