import sys

import gym
import minigrid

import sys
sys.path.insert(0, "../Minigrid")
import gym_minigrid

from gym.envs.registration import register
from gym.envs.registration import EnvSpec
from minigrid.envs.empty import EmptyEnv

# Empty
# ----------------------------------------

empty_env_5x5 = EmptyEnv(size=6)
print(empty_env_5x5.max_steps)
empty_env_5x5_spec = EnvSpec(id_requested="MiniGrid-Empty-5x5-v01",
                             max_episode_steps=empty_env_5x5.max_steps,
                             entry_point="gym_minigrid.envs:EmptyEnv",
                             kwargs={"size": 5},
                             )


# register(empty_env_5x5_spec)

register(
    id="MiniGrid-Empty-5x5-v0",
    entry_point="gym_minigrid.envs:EmptyEnv",
    max_episode_steps=empty_env_5x5.max_steps,
    kwargs={"size": 5},
)

env = gym.make("MiniGrid-Empty-5x5-v0", seed=42)

env2 = gym.make("FrozenLake-v1", is_slippery=False)

print()
