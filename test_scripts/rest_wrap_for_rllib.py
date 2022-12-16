import os
import time

import gym
import numpy as np
import pandas as pd
import ray
# from ray import tune, rllib
# from ray.rllib.algorithms.ppo import PPOConfig

from gym_minigrid.envs import EmptyEnv
from gym_minigrid.wrappers import FlatObsWrapper
from gym_minigrid.window import Window

# class T:
#     def __init__(self, mydict):
#         self.first = mydict['first']
#         self.second = mydict['second']
#
# class V(T):
#     def __init__(self, ):
#
# t = T({'first': 1, 'second': 2})
# print(t.first)
# print(t.second)



from gym.envs.toy_text import FrozenLakeEnv

class WrappedLake(FrozenLakeEnv):

    def __init__(self, env, random_arg=5):
        super().__init__(env)

        self.random_arg = random_arg

wrapped_lake = WrappedLake(FrozenLakeEnv, random_arg=10)

def myfunc(cls, *args, **kwargs):
    froze_lake = cls(*args, **kwargs)
    print(froze_lake.random_arg)
    print(froze_lake.is_slippery)

env_config = {'map_name': '4x4', 'is_slippery': False}

myfunc(wrapped_lake.__wrapped__(**env_config))