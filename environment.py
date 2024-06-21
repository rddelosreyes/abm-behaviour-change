import numpy as np
from utils import *
from agent import Agent

class Environment():
    """Describes environment parameters."""

    def __init__(self):
        self.current_timestep = 0
        self.action_list = ['A', 'B']
        self.rewarded_action = 'A'  # 'A' or 'B' only

        self.stimulus_list = []
        self.stimulus_positive_val = 1
        self.stimulus_negative_val = 0

        self.rewarded_action_list = []

    def _get_stimulus(self, agent: Agent):
        """Returns the environment stimulus."""

        if self.rewarded_action == agent.current_action:
            return self.stimulus_positive_val
        else:
            return self.stimulus_negative_val

    def _step(self, agent: Agent) -> None:
        """Simulates one step of the model."""

        self.rewarded_action_list.append(self.rewarded_action)

        agent.memory_actions.append(agent.current_action)

        # Compute stimulus associated to current and previous actions
        agent.tendency_stimulus = self._get_stimulus(agent)

        # Compute influence of memory to the choice of action
        if agent.current_action == 'A':
            sign_wrt_action = agent.sign_wrt_A
        else:
            sign_wrt_action = -np.array(agent.sign_wrt_A)

        if agent.tau_val == 0:
            agent.tendency_memory = 0
        else:
            agent.tendency_memory = function_memory_influence(agent.memory_tendency_stimulus, sign_wrt_action, agent.tau_val)

        # Compute the probability of staying with an action
        decision_val = agent.tendency_stimulus + agent.tendency_memory
        p_staying = function_logistic(decision_val, k=agent.k_val)

        # Store values
        if agent.current_action == 'A':
            agent.memory_p_A.append(p_staying)
            agent.memory_tendency_memory_A.append(agent.tendency_memory)
            agent.sign_wrt_A.append(1)
            agent.current_action = np.random.choice(self.action_list, p=[p_staying, 1-p_staying])
        else:
            agent.memory_p_A.append(1 - p_staying)
            agent.memory_tendency_memory_A.append(-agent.tendency_memory)
            agent.sign_wrt_A.append(-1)
            agent.current_action = np.random.choice(self.action_list, p=[1-p_staying, p_staying])

        agent.memory_tendency_stimulus.append(agent.tendency_stimulus)

        self.current_timestep += 1

        return None