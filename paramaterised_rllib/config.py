# TODO: currently just storing configs as dict in this .py file.
#       more elegant solutions available....

from paramaterised_rllib.envs import EmptyEnvP
from paramaterised_rllib.trainers import DummyTrainer

trainer_dict = {"DummyTrainer": DummyTrainer}
envs_dict = {"EmptyEnvP": EmptyEnvP}
env_config_defaults = {"EmptyEnvP": {"mission_space": "get to the green goal square",
                                     "grid_size": 8,
                                     "see_through_walls": True,
                                     "agent_start_pos": (1, 1),
                                     "agent_start_dir": 0}}


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