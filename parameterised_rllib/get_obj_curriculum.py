""" Example file to make a hand crafted curriculum for GoToObject environment
We make use of parameterised environment settings by changing both the
natural language mission and the physical characterstics of the environment."""
import os
from datetime import datetime as dt
import sys

import ray

sys.path.insert(0, "./")

from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune, air

from parameterised_rllib.envs import GoToObjectEnvP

env_config = {"mission_space": "go to the green ball",
              "num_objs": 2,
              "grid_size": 3,
              "see_through_walls": True, # this is hard coded in mini_grid.envs.gotoobject
              "obj_types": ["key", "ball", "box"],
              "color_names": ["red", "blue", "yellow"]
              }

# Hand crafted curriculum:
obj_types = ["key", "ball", "box"]
color_names = ["red", "blue", "yellow"]
curriculum = []
for num_obj in [1, 2]:
    for obj in obj_types:
        curriculum.append((3, num_obj, obj, ["red", "blue"]))
        curriculum.append((3, num_obj, obj, ["red", "yellow"]))
        curriculum.append((3, num_obj, obj, ["blue", "yellow"]))
    for col in color_names:
        curriculum.append((3, num_obj, ["key", "ball"], col))
        curriculum.append((3, num_obj, ["key", "box"], col))
        curriculum.append((3, num_obj, ["ball", "box"], col))
for grid_size in [4, 5, 6, 8]:
    curriculum.append((grid_size, 2, obj_types, color_names))

run = 'PPO'
stop_iters = 50
stop_timesteps = 200000
stop_reward = 0.75

level = 0
run_time = dt.now().strftime("%Y-%m-%d_%H-%M-%S")

# for level in len(curriculum):
for level in [0]:
    env_config["grid_size"] = curriculum[level][0]
    env_config["num_objs"] = curriculum[level][1]
    env_config["obj_types"] = curriculum[level][2]
    env_config["color_names"] = curriculum[level][3]

    ppo_config = (PPOConfig()
                  .environment(GoToObjectEnvP,
                               env_config=env_config)
                  .framework('torch')
                  .rollouts(num_rollout_workers=1)
                  .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", 0)))
    )

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }

    print("Training level {} with ray.tune".format(level))
    tuner = tune.Tuner(
        run,
        param_space=ppo_config.to_dict(),
        run_config=air.RunConfig(stop=stop,
                                 local_dir=f"./results/curriculum_{run_time}",
                                 name=f"level_{level}"
                                 )
    )

    results = tuner.fit()
    cp = results.get_best_result("mean_reward", mode="max").checkpoint

ray.shutdown()