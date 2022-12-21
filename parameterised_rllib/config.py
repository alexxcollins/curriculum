# TODO: currently just storing configs as dict in this .py file.
#       more elegant solutions available....

from gym_minigrid.minigrid_env import COLOR_NAMES
from parameterised_rllib.envs import EmptyEnvP, FourRoomsEnvP, GoToObjectEnvP
from parameterised_rllib.trainers import DummyTrainer

trainer_dict = {"DummyTrainer": DummyTrainer}
envs_dict = {"EmptyP": EmptyEnvP,
             "FourRoomsP": FourRoomsEnvP,
             "GoToObjectP": GoToObjectEnvP
             }
env_config_defaults = {"EmptyP": {"mission_space": "reach the goal",
                                  "grid_size": 8,
                                  "see_through_walls": True,
                                  "agent_pos": (1, 1),
                                  "agent_start_dir": 0
                                  },
                       "FourRoomsP": {"mission_space": "get to the green goal square",
                                      "agent_pos": None,
                                      "goal_pos": None,
                                      "grid_size": 19,  # this is hard coded in mini_grid.envs.fourrooms
                                      "see_through_walls": False,
                                      "max_steps": 100 # this is hard coded in mini_grid.envs.fourrooms
                                      },
                       "GoToObjectP": {"mission_space": "go to the green ball",
                                       "num_objs": 2,
                                       "grid_size": 6,
                                       "see_through_walls": True, # this is hard coded in mini_grid.envs.gotoobject
                                       "obj_types": ["key", "ball", "box"],
                                       "color_names": COLOR_NAMES,
                                       }
                       }


def get_minigrid_words():
    colors = ["red", "green", "blue", "yellow", "purple", "grey"]
    objects = [
        "unseen",
        "empty",
        "wall",
        "floor",
        "box",
        "key",
        "ball",
        "door",
        "goal",
        "agent",
        "lava",
    ]

    verbs = [
        "pick",
        "avoid",
        "get",
        "find",
        "put",
        "use",
        "open",
        "go",
        "fetch",
        "reach",
        "unlock",
        "traverse",
    ]

    extra_words = [
        "up",
        "the",
        "a",
        "at",
        ",",
        "square",
        "and",
        "then",
        "to",
        "of",
        "rooms",
        "near",
        "opening",
        "must",
        "you",
        "matching",
        "end",
        "hallway",
        "object",
        "from",
        "room",
    ]

    all_words = colors + objects + verbs + extra_words
    assert len(all_words) == len(set(all_words))
    return {word: i for i, word in enumerate(all_words)}