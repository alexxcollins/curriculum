import numpy as np
from gym_minigrid.minigrid_env import MiniGridEnv


class DummyTrainer:
    """Dummy Trainer class.

    Use its `compute_action` method to get a new action for one of the agents,
    given the agent's observation (a single discrete value encoding the field
    the agent is currently in).
    """
    def compute_action(self, env: MiniGridEnv, agent_obs=None, all_actions=False):
        # Returns a random action for a single agent.
        if all_actions:
            # if agent randomly chooses across whole space including pickup, drop, toggle then it will very likely
            # take a very long time on simple tasks and complex tasks are probably futile anyway.
            return np.random.randint(env.action_space.n - 1)  # Discrete(n) -> return rand int between 0 and n-1 (incl. n-1).
        else:
            # at least there is a chance of completing an empty room task
            return np.random.randint(3)  # Discrete(4) -> return rand int between 0 and 2 (incl. 2).