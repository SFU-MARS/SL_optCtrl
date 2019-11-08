import sys,os
sys.path.append(os.environ['PROJ_HOME_3'] + "/tests")
import numpy as np
import gym
import tests.test_vpg_off_policy.core as test_core
import tests.test_vpg_off_policy.vpg as test_vpg
from keras.models import load_model
from gym_foo import gym_foo
import pandas as pd
import pickle

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


def norm(x, stats):
    return (x - stats['mean']) / stats['std']

# ----- This is for collecting transitions for VPG replay ----- #
def save_vpg_replay(agent='quad'):
    '''
    Transitions style: 2-dim numpy array.
    [[(), (), (), ..., ()],
     [(), (), (), ..., ()]]
    '''
    assert os.path.exists(os.environ['PROJ_HOME_3'] + "/data/{}/polFunc_filled.csv".format(agent))
    dataset_path = os.environ['PROJ_HOME_3'] + "/data/{}/polFunc_filled.csv".format(agent)
    if agent == 'quad':
        env = gym.make("PlanarQuadEnv-v0", reward_type='hand_craft', set_additional_goal='None')
        # --- load pre-trained value network --- #
        value_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/quad/vf_merged.h5')
        # --- prepare original MPC dataset --- #
        mpc_column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
        raw_dataset = pd.read_csv(dataset_path, names=mpc_column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
        dataset = raw_dataset.copy()
        labelset = pd.concat([dataset.pop(x) for x in ['a1', 'a2']], 1)

        # --- prepare stats for proper normalization --- #
        val_filled_path = os.environ['PROJ_HOME_3'] + "/data/{}/valFunc_filled.csv".format(agent)
        val_filled_mpc_path = os.environ['PROJ_HOME_3'] + "/data/{}/valFunc_mpc_filled.csv".format(agent)
        assert os.path.exists(val_filled_mpc_path) and os.path.exists(val_filled_path)
        val_column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
        val_filled = pd.read_csv(val_filled_path, names=val_column_names, na_values="?", comment='\t', sep=",",
                                 skipinitialspace=True, skiprows=1)
        val_filled_mpc = pd.read_csv(val_filled_mpc_path, names=val_column_names, na_values="?", comment='\t', sep=",",
                                     skipinitialspace=True, skiprows=1)
        val_filled.pop('value')
        val_filled_mpc.pop('value')

        stats_source = pd.concat([val_filled.copy(), val_filled_mpc.copy()])
        stats_source.dropna()
        stats = stats_source.describe()
        stats = stats.transpose()

        norm_dataset = norm(dataset, stats)
        print(norm_dataset.head())

        idx = 0
        nrows = len(norm_dataset)
        # nrows = 1000
        vpg_buffer = []
        single_episode = []
        while idx < nrows:
            # ---- state input at each step ----
            state = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            norm_state = norm_dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]

            # ---- re-scale action, from [0,12] -> [-1,1]; do not use inverse sigmoid since it's nonlinear and inaccurate when encountering MPC-like data (most data is around limit) -----
            action = labelset.loc[idx, ['a1', 'a2']]
            action = [-1 + (1 - (-1)) * (a_i - 0) / (12 - 0) for a_i in action]

            # ---- reward and terminal flag at each step ----
            r = 1000 if env._in_goal(state[0:6]) else 0
            done = 1.0 if r == 1000 else 0.0

            # ---- value estimate based on value network ----
            val = value_model.predict(np.array([norm_state]))

            # ---- single transition format: (s, a, r, val, logp), but here we only save (s, a, r, val), logp is added while training. ----
            # ---- add one transition into vpg buffer. remember to add normed state because value network is pre-trained from qick_trainer.py by normed state ----
            single_episode.append((np.array(norm_state), np.array(action), r, val))

            # ---- jump over replicate terminal states  ----
            while done:
                idx += 1
                if idx < nrows:
                    state = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
                    action = labelset.loc[idx, ['a1', 'a2']]
                    r = 1000 if env._in_goal(state[0:6]) else 0
                    done = 1.0 if r == 1000 else 0.0
                    if not done:
                        idx -= 1
                        vpg_buffer.append(single_episode)
                        single_episode = []
                else:
                    break

            idx += 1  # idx += 1 should be at correct position

        print("VPG buffer:", vpg_buffer[:10])

        with open(os.environ['PROJ_HOME_3'] + "/tests/test_vpg_off_policy/vpg_transitions.pkl", 'wb') as vpg_trans:
            pickle.dump(vpg_buffer, vpg_trans)
        env.close()

if __name__ == "__main__":
    save_vpg_replay('quad')