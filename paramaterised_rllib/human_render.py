import time
import argparse
import sys
sys.path.insert(0, "./")

from gym_minigrid.window import Window
from gym_minigrid.minigrid_env import MiniGridEnv, MissionSpace

from paramaterised_rllib.config import trainer_dict, envs_dict, env_config_defaults

# TODO: these are config type files. They COULD live in a config file

class EnvRender:
    def __init__(self, Env: MiniGridEnv, Trainer, env_config, window_name: str = "gym_minigrid"):
        self.env = Env(env_config)
        self.trainer = Trainer()
        self.max_steps = 1000
        self.window = Window(window_name)

    def redraw(self, img, env=None):
        img = env.render(mode="rgb_array", tile_size=32)
        self.window.show_img(img)

    def reset(self):
        obs = self.env.reset()
        if hasattr(self.env, "mission"):
            print("Mission: %s" % self.env.mission)
            self.window.set_caption(self.env.mission)

        self.redraw(obs, self.env)
        return obs

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if done:
            print("done!")
            self.reset()
        else:
            self.redraw(obs, self.env)

        return obs, reward, done, _

    def run_one_episode(self):
        """Train an agent for one episode.

        Returns:
            episode_reward (float): The total reward for the episode.
        """
        episode_reward = 0
        episode = 0
        done = False
        obs = self.reset()

        while not done:
            action = self.trainer.compute_action(env=self.env, agent_obs=obs)
            obs, reward, done, _ = self.step(action)
            episode_reward += reward
            episode += 1
            time.sleep(0.03)

        return episode_reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="class name of environment to load", type=str, default="EmptyEnvP")
    parser.add_argument("--trainer", help="class name of trainer to load", type=str, default="DummyTrainer")
    parser.add_argument("--grid_size", help="environment size", type=int, default=None)
    parser.add_argument("--width", help="environment width", type=int, default=None)
    parser.add_argument("--height", help="environment height", type=int, default=None)
    parser.add_argument("--max_steps", help="maximum number of steps per episode", type=int, default=1000)
    # parser.add_argument("--see_through_walls", help="can agent see through walls", type=bool, default=False)
    parser.add_argument("--agent_view_size", help="agent view size", type=int, default=7)
    parser.add_argument("--highlight", help="", type=bool, default=True)
    parser.add_argument("--tile_size", help="size of each tile in pixels", type=int, default=32)
    parser.add_argument("--mission_space", help="mission string", type=str, default=None)

    args = parser.parse_args()

    # create an args dict object so we can pop items and we are left with the env_config variables
    args_dict = vars(args)
    env_name = args_dict.pop("env")
    Env = envs_dict[env_name]
    Trainer = trainer_dict[args_dict.pop("trainer")]

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
    # now add all key value pairs left in env_config_defaults to env_config
    env_config.update(env_config_defaults[env_name])

    env_render = EnvRender(Env, Trainer, env_config=env_config, window_name=env_name)
    episode_reward = env_render.run_one_episode()
    print(f"episode_reward: {episode_reward}")
