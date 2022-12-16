from gym_minigrid.envs import EmptyEnv
from gym_minigrid.wrappers import FlatObsWrapper
from gym_minigrid.wrappers import DictObservationSpaceWrapper
from gym.wrappers import FlattenObservation

env = EmptyEnv(size=6, agent_start_pos=(1,1))

env2 = FlatObsWrapper(env)

env3 = DictObservationSpaceWrapper(env, max_words_in_mission=30)
env4 = FlattenObservation(env3)

print("reset and run one step of env\n")
obs = env.reset()
print(f"obs type: {type(obs)}")
print(f"obs shape: {len(obs)}")
print(f"obs: {obs}")

print("step:")
obs, reward, done, info = env.step(0)
print(f"obs type: {type(obs)}")
print(f"obs shape: {len(obs)}")
print(f"obs: {obs}")

print()
print("reset and run one step of env2\n")
obs2 = env2.reset()
print(f"obs type: {type(obs2)}")
print(f"obs shape: {obs2.shape}")
print(f"obs: {obs2}")

print("step:")
obs2, reward2, done, info = env2.step(0)
print(f"obs type: {type(obs2)}")
print(f"obs shape: {obs2.shape}")
print(f"obs: {obs2}")

print()
print("reset and run one step of env3\n")
obs3 = env3.reset()
print(f"obs type: {type(obs3)}")
# print(f"obs shape: {obs3.shape}")
print(f"obs: {obs3}")

print("step:")
obs3, reward3, done, info = env3.step(0)
print(f"obs type: {type(obs3)}")
# print(f"obs shape: {obs.shape}")
print(f"obs: {obs}")

print('finished')