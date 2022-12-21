import numpy as np
from gym_minigrid.minigrid_env import MiniGridEnv


class DummyTrainer:
    """Dummy Trainer class.

    Use its `compute_action` method to get a new action for one of the agents,
    given the agent's observation (a single discrete value encoding the field
    the agent is currently in).
    """
    def compute_single_action(self, agent_obs=None):
        # Returns a random action for a single agent.
        # at least there is a chance of completing an empty room task if we restrict actions to just moving
        return np.random.randint(3)  # Discrete(4) -> return rand int between 0 and 2 (incl. 2).