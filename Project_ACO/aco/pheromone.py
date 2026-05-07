from .genome import AVAILABLE_ROLES, AVAILABLE_CAPABILITIES


class PheromoneMatrix:
    """
    Stores and updates pheromone trails for ACO over MAS configurations.

    For each agent position i:
      - tau_roles[i][r]     : pheromone for "agent i has role r"
      - tau_caps[i][c]      : pheromone for "agent i has capability c"

    For each directed link (i, j), i != j:
      - tau_links[i][j][v]  : pheromone for link being v=0 (off) or v=1 (on)

    Selection probability is proportional to pheromone mass (alpha=1, no heuristic).
    MMAS-style tau_min/tau_max bounds prevent trail stagnation.
    """

    def __init__(self, num_agents, tau_init=1.0, tau_min=0.01, tau_max=10.0):
        self.num_agents = num_agents
        self.tau_min = tau_min
        self.tau_max = tau_max
        n_roles = len(AVAILABLE_ROLES)
        n_caps = len(AVAILABLE_CAPABILITIES)

        self.tau_roles = [[tau_init] * n_roles for _ in range(num_agents)]
        self.tau_caps = [[tau_init] * n_caps for _ in range(num_agents)]
        # tau_links[i][j] = [tau_off, tau_on]; diagonal unused
        self.tau_links = [[[tau_init, tau_init] for _ in range(num_agents)]
                          for _ in range(num_agents)]

    def evaporate(self, rho):
        """Apply (1 - rho) decay to all pheromone values, clamped to tau_min."""
        decay = 1.0 - rho
        for i in range(self.num_agents):
            for r in range(len(AVAILABLE_ROLES)):
                self.tau_roles[i][r] = max(self.tau_min, self.tau_roles[i][r] * decay)
            for c in range(len(AVAILABLE_CAPABILITIES)):
                self.tau_caps[i][c] = max(self.tau_min, self.tau_caps[i][c] * decay)
            for j in range(self.num_agents):
                if i != j:
                    for v in range(2):
                        self.tau_links[i][j][v] = max(
                            self.tau_min, self.tau_links[i][j][v] * decay
                        )

    def deposit(self, solution, delta):
        """Add delta to pheromone trails corresponding to the choices in solution."""
        for i, agent in enumerate(solution.agents):
            r_idx = AVAILABLE_ROLES.index(agent["role"])
            c_idx = AVAILABLE_CAPABILITIES.index(agent["capability"])
            self.tau_roles[i][r_idx] = min(self.tau_max, self.tau_roles[i][r_idx] + delta)
            self.tau_caps[i][c_idx] = min(self.tau_max, self.tau_caps[i][c_idx] + delta)

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    v = solution.links[i][j]
                    self.tau_links[i][j][v] = min(
                        self.tau_max, self.tau_links[i][j][v] + delta
                    )

    def role_probs(self, agent_idx):
        tau = self.tau_roles[agent_idx]
        total = sum(tau)
        return [t / total for t in tau]

    def cap_probs(self, agent_idx):
        tau = self.tau_caps[agent_idx]
        total = sum(tau)
        return [t / total for t in tau]

    def link_prob_on(self, i, j):
        """Probability that link i->j is active."""
        tau = self.tau_links[i][j]
        return tau[1] / (tau[0] + tau[1])
