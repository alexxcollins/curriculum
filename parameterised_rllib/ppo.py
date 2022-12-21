import argparse
import os
from datetime import datetime as dt
import pickle
import sys
sys.path.insert(0, "./")

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.pre_checks.env import check_env
from ray.rllib.env.env_context import EnvContext
import ray
from ray import tune, air

from gym_minigrid.minigrid_env import MiniGridEnv, MissionSpace
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

from envs import EmptyEnvP
from config import env_config_defaults, envs_dict

def ppo_train(Env: MiniGridEnv, env_config: EnvContext, training_config: PPOConfig):
    """Train a PPO agent using Tune

    Parameters
    ----------
    env_name : str
        Name of the environment to train the agent on
    config : EnvContext
        RLlib EnvContext object containing the environment parameters
    """
    # Todo: finish re-writing this function from here
    #       #########################################
    # create environment
    env = Env(env_config)
    # obs = env.reset()

    # Check we do not have any environment errors
    print("checking environment ...")
    try:
        check_env(env)
        print("All checks passed. No errors found.")
    except:
        print("failed")

    # Create the ppo config object
    ppo_config = (PPOConfig()
                  .environment(Env,
                               env_config=env_config)
                  .framework("torch")
                  .rollouts(num_rollout_workers=1)
                  .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
                  )

    ppo_algo = ppo_config.build()

    run = training_config['run']
    no_tune = training_config['no_tune']
    as_test = training_config['as_test']
    stop_iters = training_config['stop_iters']
    stop_timesteps = training_config['stop_timesteps']
    stop_reward = training_config['stop_reward']

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
            run_config=air.RunConfig(stop=stop,
                                     # Tddo - directory needs to be relative to project folder not to folder script is run from
                                     local_dir="./results",
                                     name=f"test_experiment_{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                                     )
        )
        results = tuner.fit()

        if as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, stop_reward)

        print(results.get_best_result("mean_reward", mode="max").checkpoint)



    ray.shutdown()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--no_tune", type=bool, default=False)
    parser.add_argument("--as_test", type=bool, default=True)
    parser.add_argument("--stop_iters", type=int, default=50)
    parser.add_argument("--stop_timesteps", type=int, default=200000)
    parser.add_argument("--stop_reward", type=float, default=0.5)
    parser.add_argument("--env", help="class name of environment to load",
                        type=str, default="GoToObjectP")
    parser.add_argument("--grid_size", help="environment size", type=int, default=None)
    parser.add_argument("--width", help="environment width", type=int, default=None)
    parser.add_argument("--height", help="environment height", type=int, default=None)
    parser.add_argument("--agent_pos", help="start position of the agent", type=tuple, default=None)
    parser.add_argument("--max_steps", help="maximum number of steps per episode", type=int, default=None)
    parser.add_argument("--see_through_walls", help="can agent see through walls", type=bool, default=None)
    parser.add_argument("--agent_view_size", help="agent view size", type=int, default=7)
    parser.add_argument("--highlight", help="", type=bool, default=True)
    parser.add_argument("--tile_size", help="size of each tile in pixels", type=int, default=32)
    parser.add_argument("--mission_space", help="mission string", type=str, default=None)
    parser.add_argument("--num_objs", help="number of objects in environment",
                        type=int, default=None)
    parser.add_argument("--obj_types", help="list of object types", type=list, default=None)
    parser.add_argument("--color_names", help="list of color names", type=list, default=None)

    args = parser.parse_args()

    # create an args dict object so we can pop items and we are left with the env_config variables
    args_dict = vars(args)
    env_name = args_dict.pop("env")

    training_config = {}
    training_config["run"] = "PPO"
    training_config["no_tune"] = args_dict.pop("no_tune")
    training_config["as_test"] = args_dict.pop("as_test")
    training_config["stop_iters"] = args_dict.pop("stop_iters")
    training_config["stop_timesteps"] = args_dict.pop("stop_timesteps")
    training_config["stop_reward"] = args_dict.pop("stop_reward")

    Env = envs_dict[env_name]

    # start building env_config to pass into Env classe
    mission_string = args_dict.pop('mission_space', None)
    if mission_string is None:
        mission_string = env_config_defaults[env_name].pop('mission_space')

    env_config = {"mission_space": MissionSpace(
        mission_func=lambda: mission_string
    )}

    # set environment class specific defaults for env_config:
    for key, val in args_dict.items():
        if val is None:
            env_config[key] = env_config_defaults[env_name].pop(key, None)
        else:
            env_config[key] = val
            _ = env_config_defaults[env_name].pop(key, None) # we need to do this to avoid the config default over-riding the named arg
    # now add all key value pairs left in env_config_defaults to env_config
    _ = 1
    env_config.update(env_config_defaults[env_name])

    # save env_config and training_config to json files

    ppo_train(Env, env_config, training_config)