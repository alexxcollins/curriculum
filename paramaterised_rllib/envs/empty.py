from ray.rllib.env.env_context import EnvContext
from gym_minigrid.envs import EmptyEnv, FourRoomsEnv

from gym_minigrid.minigrid_env import MiniGridEnv


class EmptyEnvP(EmptyEnv):

    def __init__(self, env_config: EnvContext):
        try:
            env_config['max_steps'] = 4 * env_config['grid_size'] * env_config['grid_size']
            self.agent_start_pos = env_config.pop('agent_start_pos')
            self.agent_start_dir = env_config.pop('agent_start_dir')
        except KeyError:
            # raise exception do not pass
            raise KeyError('check agent_start_pos, agent_start_dir and grid_size are be specified in env_config')

        # see_through_walls in env_config will hopefully be set to True for maximum speed
        MiniGridEnv.__init__(self, **env_config)



