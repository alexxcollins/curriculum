from ray.rllib.env.env_context import EnvContext

from gym_minigrid.minigrid_env import MissionSpace
from gym_minigrid.envs import GoToObjectEnv

from gym_minigrid.minigrid_env import COLOR_NAMES
from gym_minigrid.minigrid_env import MiniGridEnv


class GoToObjectEnvP(GoToObjectEnv):
    """Create new RLlib compatible GoToObject Environment

    This class inherits from the GoToObjectEnv class in gym-minigrid and overrides the
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
            self.numObjs = env_config.pop('num_objs', 2)
            self.obj_types = env_config.pop('obj_types', ['key', 'ball', 'box'])
            self.color_names = env_config.pop('color_names', COLOR_NAMES)
        # TODO: remove the except block if code runs without error
        #       - it shouldn't ever get triggered
        except KeyError:
            raise KeyError('check num_objs are be specified in env_config')

        env_config['mission_space'] = MissionSpace(
            mission_func=lambda color, type: f"go to the {color} {type}",
            ordered_placeholders=[self.color_names, self.obj_types],
        )
        env_config['max_steps'] = (4 * env_config.get('grid_size', 8)
                                   * env_config.get('grid_size', 8)
                                   )

        MiniGridEnv.__init__(self, **env_config)
