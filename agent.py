import numpy as np

class Agent():
    """Describes agent parameters."""

    def __init__(self, tau_val: int, k_val: int):
        self.tau_val = tau_val
        self.k_val = k_val

        self.current_action = ''
        self.tendency_stimulus = 0
        self.tendency_memory = 0
        self.tendency_neighbor = 0
        self.memory_actions = []
        self.memory_tendency_stimulus = []
        self.memory_tendency_memory_A = []
        self.sign_wrt_A = []
        self.memory_p_A = []