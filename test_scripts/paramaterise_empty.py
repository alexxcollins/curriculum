import time

import numpy as np
from gym_minigrid.envs import EmptyEnv
from gym_minigrid.window import Window
import gym
print(f"gym: {gym.__version__}\n")

# TODO: turn the run_one_episode function into a class

size = 6
start_pos = (1,1)
max_steps = 100 # not used in some environments - e.g. EmptyEn

dummy_trainer = DummyTrainer()
window = Window("EmptyEnv")

def redraw(img, env=None):
    img = env.render(mode="rgb_array", tile_size=32)
    window.show_img(img)

def reset(env):
    obs = env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    redraw(obs, env)
    return obs

def step(env, action):
    obs, reward, done, _ = env.step(action)
    print(f"step={env.step_count}, reward={reward:.2f}")

    if done:
        print("done!")
        reset(env)
    else:
        redraw(obs, env)

    return obs, reward, done, _

def run_one_episode(env, trainer):
    """Train an agent for one episode.

    Returns:
        episode_reward (float): The total reward for the episode.
    """
    episode_reward = 0
    episode = 0
    done = False
    obs = reset(env)

    while not done:
        action = trainer.compute_action(obs)
        obs, reward, done, _ = step(env, action)
        episode_reward += reward
        episode += 1
        time.sleep(0.03)

    return episode_reward

if __name__ == "__main__":
    env = EmptyEnv(size=size, start_pos=start_pos)

    # Train the agent for one episode.
    episode_reward = run_one_episode(env, dummy_trainer)
    print("Episode reward: {}".format(episode_reward))

    # Start the GUI event loop.
    window.show(block=True)
