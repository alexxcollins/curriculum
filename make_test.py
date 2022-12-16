import sys

import gym
import minigrid

import gym_minigrid

from gym.envs.registration import register
from gym.envs.registration import EnvSpec
from minigrid.envs.empty import EmptyEnv

# Empty
# ----------------------------------------

# empty_env_5x5 = EmptyEnv(size=6)
# print(empty_env_5x5.max_steps)
# empty_env_5x5_spec = EnvSpec("MiniGrid-Empty-5x5-v0",
#                              max_episode_steps=empty_env_5x5.max_steps,
#                              )
#
# register(empty_env_5x5_spec, entry_point="gym_minigrid.envs:EmptyEnv",
#          kwargs={"size": 5})
# register(
#     id="MiniGrid-Empty-5x5-v0",
#     entry_point="gym_minigrid.envs:EmptyEnv",
#     max_episode_steps=empty_env_5x5.max_steps,
#     kwargs={"size": 5},
# )
# print(dir(minigrid))
# print(minigrid.__path__)
# print(minigrid.__file__)
# print(minigrid.__name__)
# print(minigrid.__package__)
# print(minigrid.__doc__)
#
# print()
# print(dir(gym_minigrid))
# print(gym_minigrid.__path__)
# print(gym_minigrid.__file__)
# print(gym_minigrid.__name__)
# print(gym_minigrid.__package__)
# print(gym_minigrid.__doc__)
#
# print()
# print(sys.path)

# env = gym.make("MiniGrid-Empty-5x5-v0", seed=42)
#
# env2 = gym.make("FrozenLake-v1", is_slippery=False)

print()
