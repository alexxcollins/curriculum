"""
Example of a curriculum learning setup using the `TaskSettableEnv` API
and the env_task_fn config.

This example shows:
  - Writing your own curriculum-capable environment using parameterised_rllib.env.
  - Defining a env_task_fn that determines, whether and which new task
    the env(s) should be set to (using the TaskSettableEnv API).
  - Using Tune and RLlib to curriculum-learn this env.

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse

import ray
from ray import air, tune
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from parameterised_rllib.curriculum_capable_env import CurriculumCapableEnv

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=200000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=10000.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv,
        env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    new_task = int(train_results["episode_reward_mean"])
    # Clamp between valid values, just in case:
    new_task = if
    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\nSetting env to task={new_task}"
    )
    return new_task