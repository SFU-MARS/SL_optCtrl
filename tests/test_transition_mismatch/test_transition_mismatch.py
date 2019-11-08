# code for testing transitive probability mismatch between mpc and gazebo
# [1] randomly select an starting point from mpc trajectories
# [2] apply subsequent actions from the starting state
# [3] see the errors both visually and statistically. (how many steps agent can last)
# remember to set same control step size between Gazebo and MPC: 0.05s
# Gazebo world file: max_step_size

# Conclusion: after around 7 steps, the shift error is unacceptable.
import os,sys
import numpy as np
import pandas as pd

import gym
from gym_foo import gym_foo

agent = 'quad'
mpc_datapath = os.environ['PROJ_HOME_3'] + "/data/{}/polFunc.csv".format(agent)
assert os.path.exists(mpc_datapath)


mpc_column_names = ['samp', 'x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2']
mpc_data = pd.read_csv(mpc_datapath, names=mpc_column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)

dataset = mpc_data.copy()
dataset.pop('samp')
labelset = pd.concat([dataset.pop(x) for x in ['a1', 'a2']], 1)

starting_indices = np.arange(0,150) * 100
# print("start indices:", start_indices)

# sample_start_idx = np.random.choice(starting_indices)
sample_start_idx = 0
starting_state = dataset.loc[sample_start_idx]
print("starting state:", np.array(starting_state))

env = gym.make("PlanarQuadEnv-v0", reward_type='hand_craft', set_additional_goal='None')
env.customized_reset = [np.array(starting_state)]
o = env.reset()

for step in range(10):
    a = labelset.loc[sample_start_idx + step, ['a1', 'a2']]
    print("a:", a)
    nx_o, _, _, _ = env.step(a)
    print("next state:", nx_o)

env.close()



