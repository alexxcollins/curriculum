from ray.rllib.env.env_context import EnvContext
from gym_minigrid.envs import EmptyEnv

from gym_minigrid.minigrid_env import MiniGridEnv


class EmptyEnvP(EmptyEnv):
    """Create new RLlib compatible Empty Environment

    This class inherits from the Empty class in gym-minigrid and overrides the
    __init__ method to allow for the use of a single RLlib EnvContext object to
    define the environment parameters.

    Parameters
    ----------
    env_config : EnvContext
        RLlib EnvContext object containing the environment parameters
        It is input as a dictionary. In the original minigrid environment, keyword
        argument default values are set in the __init__ method. In this class, the
        default values are set in the env_config_defaults dictionary in config.py.
    """
    def __init__(self, env_config: EnvContext):
        try:
            env_config['max_steps'] = (4 * env_config.get('grid_size', 8)
                                       * env_config.get('grid_size', 8)
                                       )
            self.agent_start_pos = env_config.pop('agent_pos')
            self.agent_start_dir = env_config.pop('agent_start_dir')
        except KeyError:
            raise KeyError('check agent_start_pos, agent_start_dir and grid_size are be specified in env_config')

        # see_through_walls in env_config will hopefully be set to True for maximum speed
        MiniGridEnv.__init__(self, **env_config)
