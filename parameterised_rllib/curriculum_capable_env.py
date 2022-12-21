"""An example of setting up a simple curriculum learning experiment.
The curricula is handcrafted in this case, but takes advantage of the
environment parameterisation API to allow for easy experimentation with
different curricula."""

import gym
import random

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override

from parameterised_rllib.envs import GoToObjectEnvP

env_config = {"start_level": 1,
              "mission_space": "go to the green ball",
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



class CurriculumCapableEnv(TaskSettableEnv):
    """Example of a curriculum learning capable env.

    This simply wraps a GoToObj env and makes it harder with each
    task. Task (difficulty levels) can range from 1 to 10."""

    def __init__(self, config: EnvContext):
        self.cur_level = config.pop("start_level", 1)
        self.env_config = config
        self.env = None
        self._make_env()  # create self.env
        self.switch_env = False
        self._timesteps = 0

    def reset(self):
        if self.switch_env:
            self.switch_env = False
            self._make_env()
        self._timesteps = 0
        return self.env.reset()

    def step(self, action):
        self._timesteps += 1
        obs, reward, done, info = self.frozen_lake.step(action)
        reward = 10 * self.cur_level
        if self._timesteps >= self.env.max_steps:
            done = True
        return obs, reward, done, info

    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [random.randint(1, 10) for _ in range(n_tasks)]

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_level

    @override(TaskSettableEnv)
    def set_task(self, task):
        """Implement this to set the task (curriculum level) for this env."""
        self.cur_level = task
        self.switch_env = True

    def _make_env(self):
        # need to change config to make the env harder
        self.env_config["grid_size"] = curriculum[self.cur_level - 1][0]
        self.env_config["num_objs"] = curriculum[self.cur_level - 1][1]
        self.env_config["obj_types"] = curriculum[self.cur_level - 1][2]
        self.env_config["color_names"] = curriculum[self.cur_level - 1][3]
        self.env = GoToObjectEnvP(self.env_config)
