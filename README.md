# Glue and complete solution to use environment from [MiniGrid](https://minigrid.farama.org) with [Ray](https://docs.ray.io/en/latest/ray-overview/index.html)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

What they say about MiniGrid: There are other gridworld Gym environments out there, but this one is
designed to be particularly simple, lightweight and fast. The code has very few
dependencies, making it less likely to break or fail to install. It loads no
external sprites/textures, and it can run at up to 5000 FPS on a Core i7
laptop, which means you can run your experiments faster. 

Ray and the RLlib library provide complete implementations of popular RL algorithms, allow effortless scaling from laptop to cloud and provide efficient hyperparameter tuning with a variety of search methods.

Compatibility between [OpenAI Gym](https://github.com/openai/gym), Minigrid and Ray[RLlib] is difficult, particulary if you want to generate many parameterised MiniGrid Environments to use as part of a curriculum. This repo will help take the pain out of setting up with Google Cloud (GCP), including $300 of credits which get you a long way into your first experiments, and managing the dependencies between Ray and Minigrid. It has forked MiniGrid and modified the base MiniGridEnv class as well as written a lean subclass of all the MiniGridEnvs (so the MiniGrid EmptyEnv is called EmptyEnvP in this library). You can pass these Env classes into a ray AlgorithmConfig object along with an env_config dictionary to algorithmically define your environments on the fly or as part of an environment buffer.

Requirements:
- Python 3.7 to 3.10 (Python 3.8 recommended)
- OpenAI Gym v0.21
- NumPy 1.18+
- Matplotlib (optional, only needed for display) - 3.0+
- Ray[RLlib] 2.1.0
The provided environment yaml files take all the pain away from dependency management. It is highly advised that you use the enivoronment_deb.yml (for linux) or the environment_m1.yml (for mac) to build your environment.

```
conda create -n minigrid python=3.8
conda activate minigrid
conda env update --file environment_deb.yml # use m1 for mac
git clone https://github.com/alexxcollins/curriculum.git
cd gym_minigrid
pip install -e .
cd ..
```

For Mac users with Arm64 architecture (M1 or M2 chips), it is highly advisable to do the following:
* [uninstall conda](https://docs.anaconda.com/anaconda/install/uninstall/)
* You can preserve existing environments by running `conda env export --file environment.yml`
* [install homebrew](https://brew.sh)
* [install mamba](https://mamba.readthedocs.io/en/latest/) Mamba works just like conda (you can even use the same commands) but is faster and solves many problems which have accrued to conda and conda-forge over time. 
* Now create the environment above using those commands or substituting `mamba` for `conda`




