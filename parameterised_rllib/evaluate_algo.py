import argparse
import sys
sys.path.insert(0, "trainers/")

from ray.rllib.algorithms.algorithm import Algorithm

from gym_minigrid.minigrid_env import MiniGridEnv
from parameterised_rllib.config import envs_dict
from parameterised_rllib.human_render import EnvRender
from parameterised_rllib.envs import GoToObjectEnvP

def evaluate_algo(checkpoint_path: str, Env: MiniGridEnv,
                  render: bool = False):
    """Evaluate a trained RLlib algorithm

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint to evaluate
    render : bool, optional
        Whether to render the environment, by default False
    """
    algo = Algorithm.from_checkpoint(checkpoint_path)
    env = GoToObjectEnvP(algo.config["env_config"])

    if render:
        env_render = EnvRender(Env, algo, algo.config["env_config"],
                               window_name="gym_minigrid")
        env_render.run_one_episode()

    else:
        # compute actions until done
        done = False
        env = GoToObjectEnvP(algo.config["env_config"])
        obs = env.reset()
        episode_reward = 0

        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        print(episode_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="class name of environment to load",
                        type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    Env = envs_dict[args.env]
    evaluate_algo(args.checkpoint_path, Env, args.render)