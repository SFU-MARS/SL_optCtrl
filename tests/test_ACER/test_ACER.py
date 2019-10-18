import numpy as np
import os
import csv
import time
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matlab.engine
import pickle
import torch

import gym
from gym_foo import gym_foo

sys.path.append(os.environ['PROJ_HOME_3']+'/acer')
print(sys.path)

import replay_memory

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

# ---------- This function is mainly used for collecting transitions on ACER algorithm --------
def save_for_ACER_replay(agent='quad'):
    assert os.path.exists(os.environ['PROJ_HOME_3'] + "/data/quad/polFunc_filled.csv")
    dataset_path = os.environ['PROJ_HOME_3'] + "/data/quad/polFunc_filled.csv"
    if agent == 'quad':
        env = gym.make("PlanarQuadEnv-v0")
        env.reward_type = "hand_craft"
        env.set_additional_goal = 'None'

        column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
        dataset = raw_dataset.copy()
        labelset = pd.concat([dataset.pop(x) for x in ['a1', 'a2']], 1)

        idx = 0
        # nrows = len(dataset)
        nrows = 100
        transitions = []
        while idx < nrows-1:
            state = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            # ---- re-scale action, from [0,12] -> [-1,1] ------
            # ---- do not use inverse sigmoid since it's nonlinear and inaccurate when encountering MPC-like data (most data is around limit) ----
            action = labelset.loc[idx, ['a1', 'a2']]
            action = [-1 + (1 - (-1)) * (a_i - 0) / (12 - 0) for a_i in action]
            # --------------------------------------------------
            next_state = dataset.loc[idx+1, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            r = 1000 if env._in_goal(state[0:6]) else 0
            done = 1.0 if r == 1000 else 0.0
            # ---- statistics format: [mean, logsd] -----
            logsd = -0.69
            exploration_statistics = [action[0] - np.random.normal()*np.exp(logsd), action[1] - np.random.normal()*np.exp(logsd), logsd, logsd]
            idx += 1 # idx += 1 should be at correct position

            transition = replay_memory.Transition(states=torch.FloatTensor(state).view(1, -1),
                                                  actions=torch.FloatTensor(action).view(1, -1),
                                                  rewards=torch.FloatTensor([[r]]),
                                                  next_states=torch.FloatTensor(next_state).view(1, -1),
                                                  done=torch.FloatTensor([[done]]),
                                                  exploration_statistics=torch.FloatTensor(exploration_statistics).view(1,-1))
            transitions.append(transition)
            print("transition:", transition)
            while done:
                idx += 1
                if idx < nrows - 1:
                    state = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
                    action = labelset.loc[idx, ['a1', 'a2']]
                    next_state = dataset.loc[idx + 1, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
                    r = 1000 if env._in_goal(state[0:6]) else 0
                    done = 1.0 if r == 1000 else 0.0
                else:
                    break
        print("length of valid transitions:", len(transitions))
        with open(os.environ['PROJ_HOME_3'] + "/tests/test_ACER/transitions.pkl", 'wb') as trans_file:
            pickle.dump(transitions, trans_file)
        env.close()

def load_replay():
    with open(os.environ['PROJ_HOME_3'] + "/tests/test_ACER/transitions.pkl", 'rb') as trans_file:
        transitions = pickle.load(trans_file)
        for i in range(100):
            print("transition:", transitions[i])


if __name__ == "__main__":
    save_for_ACER_replay(agent='quad')
    # load_replay()
