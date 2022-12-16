import os
import time
import sys
sys.path.insert(0, "../Minigrid")

import gym
import numpy as np

from gym_minigrid.envs import EmptyEnv
from gym_minigrid.wrappers import FlatObsWrapper
from gym_minigrid.window import Window
from ray.rllib.algorithms.ppo import PPOConfig

from paramaterise_empty import DummyTrainer, redraw, reset, step, run_one_episode

print(f'Number of CPUs in this system: {os.cpu_count()}')
import pandas as pd

print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")

import ray
from ray import tune, rllib, air
from ray.tune.logger import pretty_print

print(f"ray: {ray.__version__}")

size = 6
start_pos = (1,1)
max_steps = 1000

###########################
# define the environment using EmptyEnv object
###########################
env = EmptyEnv(size=size)


if isinstance(env, gym.Env):
    print("This is a gym.Env")
    print()

    if isinstance(env.action_space, gym.spaces.Space):
        print(f"gym action space: {env.action_space}")
    if isinstance(env.observation_space, gym.spaces.Space):
        print(f"gym observation space: {env.observation_space}")
    print()

from ray.rllib.utils.pre_checks.env import check_env

# How to check you do not have any environment errors
print("checking environment ...")
try:
    check_env(env)
    print("All checks passed. No errors found.")
except:
    print("failed")

# Calculate environment baseline
print("calculating baseline ...")
num_episodes = 50
num_time_steps = 0
episode_rewards = []

# dummy_trainer = DummyTrainer()
# for ep in range(num_episodes):
#
#     obs = env.reset()
#     episode_reward = 0
#     episode_time_steps = 0
#     done = False
#
#     while not done:
#         action = dummy_trainer.compute_action(env=env, single_agent_obs=obs)
#         obs, reward, done, _ = env.step(action)
#         episode_reward += reward
#         episode_time_steps += 1
#
#     episode_rewards.append(episode_reward)
#     num_time_steps += episode_time_steps
#     # print(f"Episode {ep}: reward: {episode_reward:.2f}: episode_time_steps: {episode_time_steps}")
#
# env_mean_random_reward = np.mean(episode_rewards)
# env_sd_random_reward = np.std(episode_rewards)
# # calculate number of wins
# num_wins = np.sum(np.array(episode_rewards) > 0)
#
# print("\nBaseline results:")
# print(f"mean random reward: {env_mean_random_reward:.2f}")
# print(f"sd random reward: {env_sd_random_reward:.2f}")
# print(f"num wins: {num_wins}. winning percent: ({num_wins/num_episodes*100:.2f}%)")
# print()

# Now use rllib and PPO to train an agent
# define the environment using EmptyEnv object
from gym_minigrid.wrappers import DictObservationSpaceWrapper
from gym.wrappers import FlattenObservation

class DictObs(EmptyEnv):
    def __init__(self, env_config):
        super().__init__(**env_config)

class FlatObs(DictObs):
    def __init__(self, env_config, max_words_in_mission=30):
        super().__init__(**env_config, max_words_in_mission=max_words_in_mission)


class EmptyEnvWrapper(FlatObs):
    def __init__(self, env_config):
        try:
            self.size = env_config.pop('size')
        except KeyError:
            pass
        super().__init__(**env_config)
        self.size = env_config['size']

    # def step(self, action):
    #     obs, reward, done, _ = super().step(action)
    #     return obs, reward, done, _
    #
    # def reset(self):
    #     obs = super().reset()
    #     return obs

ppo_config_basic = PPOConfig() # create config object
ppo_config_basic.environment(env="MiniGrid-Empty-5x5-v0") # for testing just set the environment
# dnon't adjust any other config parameters
ppo_algo_basic = ppo_config_basic.build()

envt1 = EmptyEnvWrapper(env_config={"size": size, "agent_start_pos": start_pos})
# envt2 = EmptyEnv(size=size, agent_start_pos=start_pos)

ppo_config = PPOConfig()

# set up config object to use our environment
ppo_config.environment(env=EmptyEnvWrapper, env_config={
    "env_config": {'size': size, 'agent_start_pos': start_pos}
})

ppo_config2 = PPOConfig()
ppo_config2.environment(env="FrozenLake-v1")
env2 = gym.make("FrozenLake-v1")

ppo_config3 = PPOConfig()
ppo_config3.environment(env="MiniGrid-Empty-5x5-v0")
env3 = FlatObsWrapper(gym.make("MiniGrid-Empty-5x5-v0"))

# # use pytorch as the framework
# ppo_config.framework("torch")
#
# # set up evaluation
# # Setup evaluation
# ppo_config.evaluation(
#
#     # Minimum number of training iterations between evaluations.
#     # Evaluations are blocking operations (if evaluation_parallel_to_training=False)
#     # set `evaluation_interval` larger for faster runtime.
#     evaluation_interval=15,
#
#     # Minimum number of evaluation iterations.
#     # If using multiple evaluation workers, we will run at least
#     # this many episodes * num_evalworkers total.
#     evaluation_duration=5,
#
#     # Number of parallel evaluation workers.
#     # Zero by default, which means evaluation will run on the training resources.
#     # If you increase this, it will increase total Ray resource usage
#     # since evaluation workers are created separately from rollout workers
#     # Note: these show up on Ray Dashboard as extra "RolloutWorker"s
#     evaluation_num_workers=7,  #0 for Colab
#     # evaluation_num_workers=0,  # 0 for Colab
#
#     # Use the parallel evaluation workers in parallel with training workers
#     evaluation_parallel_to_training=True,  # False for Colab
#
#     evaluation_config=dict(
#         # Explicitly set "explore"=False to override default True
#         # Best practice value is False unless environment is stochastic
#         explore=False,
#
#         # Number of parallel Training workers
#         # Override the num_workers from the training config
#         # Note: ppo only allows 1 Trainer worker, see documentation
#         num_workers=1,  # any number here will be reset = 1 for ppo
#     ),
# )
#
# # Setup sampling rollout workers for streaming the data
# ppo_config.rollouts(
#     # num_rollout_workers=7,  #1 for Colab
#     num_rollout_workers=1,  # 1 for Colab
#
#     # for small environments this can be >1 based on size of your processor
#     num_envs_per_worker=1, )
#
print(f"Config type: {type(ppo_config)}")

# Use the config object's `build()` method for instantiating
# an RLlib Algorithm instance that we can then train.
# Note if using Tune, don't need algo object, but this is still a good debugging step.
# ppo_algo2 = ppo_config2.build()
# print('built Frozen Lake ppo\n')

ppo_algo3 = ppo_config3.build()
print('built make(EmptyEnv) ppo\n')


ppo_algo = ppo_config.build()
print(f"Algorithm type: {type(ppo_algo)}")