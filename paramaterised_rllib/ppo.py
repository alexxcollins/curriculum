from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.pre_checks.env import check_env
import ray
from ray import tune, air
import gym

from gym_minigrid.minigrid_env import MiniGridEnv, MissionSpace
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

from envs import EmptyEnvP
from config import env_config_defaults

env_string = "EmptyEnvP"
env_config_defaults = env_config_defaults[env_string]
env_config_defaults["mission_space"] = MissionSpace(
    mission_func=lambda: env_config_defaults["mission_space"]
)
env = EmptyEnvP(env_config=env_config_defaults.copy())
obs = env.reset()
# env2 = gym.make("MiniGrid-Empty-8x8-v0")

# Check we do not have any environment errors
print("checking environment ...")
try:
    check_env(env)
    print("All checks passed. No errors found.")
except:
    print("failed")

# print("checking environment ...")
# try:
#     check_env(env2)
#     print("All checks passed. No errors found.")
# except:
#     print("failed")

print()
# Create the ppo config object
ppo_config = (PPOConfig()
              .environment(EmptyEnvP,
                           env_config=env_config_defaults)
              .framework("torch")
              .rollouts(num_rollout_workers=1)
              .resources(num_gpus=0) # todo num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
              )

ppo_algo = ppo_config.build()

run = "PPO"
no_tune = False
as_test = True
stop_iters = 50
stop_timesteps = 100000
stop_reward = 0.5

stop = {
    "training_iteration": stop_iters,
    "timesteps_total": stop_timesteps,
    "episode_reward_mean": stop_reward,
}

if no_tune:
    # manual training with train loop using PPO and fixed learning rate
    print("Running manual train loop without Ray Tune.")
    # use fixed learning rate instead of grid search (needs tune)
    ppo_config.lr = 1e-3
    ppo_algo = ppo_config.build()
    # run manual training loop and print results after each iteration
    for _ in range(stop_iters):
        result = ppo_algo.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if (
                result["timesteps_total"] >= stop_timesteps
                or result["episode_reward_mean"] >= stop_reward
        ):
            break
    ppo_algo.stop()
else:
    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    tuner = tune.Tuner(
        run,
        param_space=ppo_config.to_dict(),
        run_config=air.RunConfig(stop=stop),
    )
    results = tuner.fit()

    if as_test:
        print("Checking if learning goals were achieved")
        check_learning_achieved(results, stop_reward)

ray.shutdown()