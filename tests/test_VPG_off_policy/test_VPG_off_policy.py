import sys,os
sys.path.append(os.environ['PROJ_HOME_3'] + "/tests")
import numpy as np
import gym
import tests.test_VPG_off_policy.core as test_core
import tests.test_VPG_off_policy.vpg as test_vpg
from keras.models import load_model
from gym_foo import gym_foo
import pandas as pd
import pickle

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

# ----- This is for collecting transitions for VPG replay ----- #
def save_VPG_replay(agent='quad'):
    '''
    Transitions style: 2-dim numpy array.
    [[(), (), (), ..., ()],
     [(), (), (), ..., ()]]
    '''
    assert os.path.exists(os.environ['PROJ_HOME_3'] + "/data/quad/polFunc_filled.csv")
    dataset_path = os.environ['PROJ_HOME_3'] + "/data/quad/polFunc_filled.csv"
    if agent == 'quad':
        env = gym.make("PlanarQuadEnv-v0")
        env.reward_type = "hand_craft"
        env.set_additional_goal = 'None'
        value_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/quad/vf.h5')

        column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
        dataset = raw_dataset.copy()
        labelset = pd.concat([dataset.pop(x) for x in ['a1', 'a2']], 1)

        idx = 0
        # nrows = len(dataset)
        nrows = 1000
        VPG_buffer = []
        single_episode = []
        while idx < nrows:
            # ---- state input at each step ----
            state = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            # ---- re-scale action, from [0,12] -> [-1,1]; do not use inverse sigmoid since it's nonlinear and inaccurate when encountering MPC-like data (most data is around limit) -----
            action = labelset.loc[idx, ['a1', 'a2']]
            action = [-1 + (1 - (-1)) * (a_i - 0) / (12 - 0) for a_i in action]
            # ---- reward and terminal flag at each step ----
            r = 1000 if env._in_goal(state[0:6]) else 0
            done = 1.0 if r == 1000 else 0.0
            # ---- value estimate based on value network ----

            tmp_s = np.array([state])
            # print(tmp_s)
            # print(tmp_s.shape)
            val = value_model.predict(tmp_s)

            # ---- statistics format: [mean, logsd] -----
            logsd = -0.69
            mean  = [action[0] - np.random.normal()*np.exp(logsd), action[1] - np.random.normal()*np.exp(logsd)]
            logp_presum = -0.5 * (((np.array(action) - np.array(mean)) / (np.exp(logsd) + 1e-8)) ** 2 + 2 * logsd + np.log(2 * np.pi))
            logp = np.sum(logp_presum)
            # ---- add one transition into VPG buffer ----
            single_episode.append((np.array(state), np.array(action), r, val, logp))

            while done:
                idx += 1
                if idx < nrows:
                    state = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
                    action = labelset.loc[idx, ['a1', 'a2']]
                    r = 1000 if env._in_goal(state[0:6]) else 0
                    done = 1.0 if r == 1000 else 0.0
                    if not done:
                        idx -= 1
                        VPG_buffer.append(single_episode)
                        single_episode = []
                else:
                    break

            idx += 1  # idx += 1 should be at correct position

        print("VPG buffer:", VPG_buffer)

        with open(os.environ['PROJ_HOME_3'] + "/tests/test_VPG_off_policy/vpg_transitions.pkl", 'wb') as vpg_trans:
            pickle.dump(VPG_buffer, vpg_trans)
        env.close()


if __name__ == "__main__":
    if not os.path.exists(os.environ['PROJ_HOME_3'] + "/tests/test_VPG_off_policy/vpg_transitions.pkl"):
        print("saving VPG transitions")
        save_VPG_replay()
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='PlanarQuadEnv-v0')
        parser.add_argument('--hid', type=int, default=64)
        parser.add_argument('--l', type=int, default=2)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--seed', '-s', type=int, default=0)
        parser.add_argument('--cpu', type=int, default=1)
        parser.add_argument('--steps', type=int, default=1024)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--exp_name', type=str, default='vpg')
        parser.add_argument('--off_policy_update', type=str, default='true')
        args = parser.parse_args()

        mpi_fork(args.cpu)  # run parallel code with mpi

        from spinup.utils.run_utils import setup_logger_kwargs

        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

        print("My VPG")

        test_vpg.vpg(lambda: gym.make(args.env, reward_type='hand_craft', set_additional_goal='None'), actor_critic=test_core.mlp_actor_critic_val_init,
            ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
            seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
            logger_kwargs=logger_kwargs, off_policy_update=args.off_policy_update)






