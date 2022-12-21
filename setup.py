from setuptools import setup

with open("README.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("##"):
            header_count += 1
        if header_count < 2:
            long_description += line
        else:
            break

setup(
    name="gym_minigrid",
    author="Alex Collins forked from Farama Foundation",
    author_email="tbc",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    version="0.1",
    keywords="memory, environment, agent, rl, gym",
    url="https://github.com/alexxcollins/curriculum.git",
    description="Paramaterised gridworld reinforcement learning environments for rllib and cloud",
    packages=["gym_minigrid", "gym_minigrid.envs"],
    entry_points={
        "gym.envs": ["__root__ = gym_minigrid.__init__:register_minigrid_envs"]
    },
    license="Apache",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.0",
    ],
    python_requires=">=3.7",
)
