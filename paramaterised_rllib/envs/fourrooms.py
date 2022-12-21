from ray.rllib.env.env_context import EnvContext
from gym_minigrid.envs import FourRoomsEnv

from gym_minigrid.minigrid_env import MiniGridEnv

class FourRoomsEnvP(FourRoomsEnv):
    """Create new RLlib compatible Four Rooms Environment

    This class inherits from the FourRoomsEnv class in gym-minigrid and overrides the
    __init__ method to allow for the use of a single RLlib EnvContext object to
    define the environment parameters.

    Parameters
    ----------
    env_config : EnvContext
        RLlib EnvContext object containing the environment parameters
        It is input as a dictionary. In the originan minigrid environment, keyword
        argument default values are set in the __init__ method. In this class, the
        default values are set in the env_config_defaults dictionary in config.py.
    """
    def __init__(self, env_config: EnvContext):
        try:
            self._agent_default_pos = env_config.pop('agent_pos')
            self._goal_default_pos = env_config.pop('goal_pos')
        except KeyError:
            raise KeyError('check agent_pos and goal_pos are be specified in env_config')

        MiniGridEnv.__init__(self, **env_config)