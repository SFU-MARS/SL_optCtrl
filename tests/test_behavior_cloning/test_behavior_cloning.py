# # -------------------- generate our MPC data with no additional successful steps ----------------------
# import os
# import pandas as pd
# import numpy as np
# import gym
# from gym_foo import gym_foo
#
# dataset_path = os.environ['PROJ_HOME_3'] + "/data/quad/polFunc_filled.csv"
# column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
# raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",",
#                           skipinitialspace=True, skiprows=1)
# dataset = raw_dataset.copy()
# labelset = pd.concat([dataset.pop(x) for x in ['a1', 'a2']], 1)
#
#
# idx = 0
# nrows = len(dataset)
# reward_sum = 0
# env = gym.make("PlanarQuadEnv-v0")
# env.reward_type = "hand_craft"
# env.set_additional_goal = 'None'
#
# episode_returns = []
# episode_starts = []
# rewards = []
# actions = []
# obs = []
# episode_starts.append(True)
# while idx < nrows-1:
#
#     cur_obs = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
#     action = labelset.loc[idx, ['a1', 'a2']]
#     # re-scale action, from [0,12] -> [-1,1]
#     action = [-1 + (1 - (-1)) * (a_i - 0) / (12 - 0) for a_i in action]
#     next_obs = dataset.loc[idx+1, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
#     r = 1000 if env._in_goal(next_obs[0:6]) else 0
#     done = True if r == 1000 else False
#
#     reward_sum += r
#     obs.append(cur_obs)
#     actions.append(action)
#     rewards.append(r)
#     episode_starts.append(done)
#
#     if done:
#         episode_returns.append(reward_sum)
#         reward_sum = 0
#
#     while done and idx < nrows-1:
#         cur_obs = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
#         action = labelset.loc[idx, ['a1', 'a2']]
#         next_obs = dataset.loc[idx + 1, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
#         r = 1000 if env._in_goal(next_obs[0:6]) else 0
#         done = True if r == 1000 else False
#         idx += 1
#
#     idx += 1
#
# episode_starts = episode_starts[:-1]
#
# expert_trajs = {}
# expert_trajs['actions'] = np.array(actions)[0:-1:10,:]
# expert_trajs['rewards'] = np.array(rewards)[0:-1:10]
# expert_trajs['episode_returns'] = np.array(episode_returns)
# expert_trajs['obs'] = np.array(obs)[0:-1:10,:]
# expert_trajs['episode_starts'] = np.array(episode_starts)[0:-1:10]
#
#
#
# print("shape of actions:", expert_trajs['actions'].shape)
# print("shape of rewards:", expert_trajs['rewards'].shape)
# print("shape of episode_returns:", expert_trajs['episode_returns'].shape)
# print("shape of obs:", expert_trajs['obs'].shape)
# print("shape of starts:", expert_trajs['episode_starts'].shape)
#
# np.savez("full_MPC_quad", **expert_trajs)
# env.close()
# # ------------------------------------------------------------------------------------------------
#
# # -------------------- generate full MPC data ----------------------
# import os
# import pandas as pd
# import numpy as np
# import gym
# from gym_foo import gym_foo
#
# dataset_path = os.environ['PROJ_HOME_3'] + "/data/quad/polFunc_filled.csv"
# column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
# raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",",
#                           skipinitialspace=True, skiprows=1)
# dataset = raw_dataset.copy()
# labelset = pd.concat([dataset.pop(x) for x in ['a1', 'a2']], 1)
#
#
# idx = 0
# nrows = len(dataset)
# reward_sum = 0
# env = gym.make("PlanarQuadEnv-v0")
# env.reward_type = "hand_craft"
# env.set_additional_goal = 'None'
#
# obs = dataset.copy()
# actions = labelset.copy()
# # re-scale action, from [0,12] -> [-1,1]
# actions = -1 + (1 - (-1)) * (actions - 0) / (12 - 0)
# rewards = [1000 if env._in_goal(np.array(obs.iloc[idx][0:6])) else 0 for idx in range(nrows)]
# episode_starts = [True if r == 1000 else False for r in rewards]
# episode_returns = [sum(rewards[idx:idx+200]) for idx in range(0,nrows,200)]
#
#
# expert_trajs = {}
# expert_trajs['actions'] = np.array(actions)
# expert_trajs['rewards'] = np.array(rewards)
# expert_trajs['episode_returns'] = np.array(episode_returns)
# expert_trajs['obs'] = np.array(obs)
# expert_trajs['episode_starts'] = np.array(episode_starts)
#
# print("shape of actions:", expert_trajs['actions'].shape)
# print("shape of rewards:", expert_trajs['rewards'].shape)
# print("shape of episode_returns:", expert_trajs['episode_returns'].shape)
# print("shape of obs:", expert_trajs['obs'].shape)
# print("shape of starts:", expert_trajs['episode_starts'].shape)
#
# np.savez("full_MPC_quad", **expert_trajs)
# env.close()
# # ------------------------------------------------------------------------------------------------

# # ----------------- pretrain stable-baselines PPO1 and save model ----------------------------------
# import gym
# from gym_foo import gym_foo
#
# from stable_baselines import PPO1
# from stable_baselines.gail import ExpertDataset
# # Using only one expert trajectory
# # you can specify `traj_limitation=-1` for using the whole dataset
# dataset = ExpertDataset(expert_path="MPC_quad_no_additional_done.npz",
#                         traj_limitation=-1, batch_size=128)
#
# model = PPO1('MlpPolicy', 'PlanarQuadEnv-v0', verbose=1)
# # Pretrain the PPO2 model
# model.pretrain(dataset, n_epochs=2500)
#
#
# model.save('ppo1_pretrain_no_additional_done')
# # # ---------------------------------------------------------------------------------

# As an option, you can train the RL agent
# model.learn(int(1e5))

# ------------------- test pretrained model on simulator -----------------------
# Test the pre-trained model
import gym
from gym_foo import gym_foo
import numpy as np
from stable_baselines.gail import ExpertDataset




# # ---------------- Test on our PPO1 --------------------
# import os
# import ppo
# import json
# full_MPC_data = np.load("full_MPC_quad.npz")
# initial_obs_pool = full_MPC_data['obs'][110:-1:200,0:6].tolist()
# env = gym.make("PlanarQuadEnv-v0")
# env.reward_type = "hand_craft"
# env.set_additional_goal = 'None'
# env.customized_reset = initial_obs_pool[np.random.choice(len(initial_obs_pool))]
#
# print("sampling source:", initial_obs_pool)
# obs = env.reset()
# print("initial obs:", obs)

# # Initialize policy
# ppo.create_session()
# init_policy = ppo.create_policy('pi', env, vf_load=False, pol_load=True)
# ppo.initialize()
# pi = init_policy
#
# # init params
# with open(os.environ['PROJ_HOME_3']+'/ppo_params.json') as params_file:
#     d = json.load(params_file)
#     num_iters = d.get('num_iters')
#     num_ppo_iters = d.get('num_ppo_iters')
#     timesteps_per_actorbatch = d.get('timesteps_per_actorbatch')
#     clip_param = d.get('clip_param')
#     entcoeff = d.get('entcoeff')
#     optim_epochs = d.get('optim_epochs')
#     optim_stepsize = d.get('optim_stepsize')
#     optim_batchsize = d.get('optim_batchsize')
#     gamma = d.get('gamma')
#     lam = d.get('lam')
#     max_iters = num_ppo_iters
#
#     for _ in range(10):
#         _, _, eval_ep_mean_reward, eval_suc_percent, _, _ = ppo.ppo_eval(env, pi, timesteps_per_actorbatch // 2,
#                                                                                max_iters=5, stochastic=False)
# env.close()
# # -------------------------------------------------------

# ----------------- Test on stable-baselines PPO1 --------------------
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

full_MPC_data = np.load("full_MPC_quad.npz")
key_points = [150, 120, 90, 60, 30, 0]
traj_radius = 5


# initial_obs_pool = full_MPC_data['obs'][120:-1:200,0:6].tolist()

env = gym.make("PlanarQuadEnv-v0")
env.reward_type = "hand_craft"
env.set_additional_goal = 'None'
# env.customized_reset = initial_obs_pool[np.random.choice(len(initial_obs_pool))]

# print("sampling source:", initial_obs_pool)
# obs = env.reset()
# print("initial obs:", obs)


reward_sum = 0.0
model = PPO1.load('ppo1_pretrain_no_additional_done')
model.env = env

# print(dir(model))


old_obs_pool = None
for kp in key_points:
    initial_obs_pool = []
    for r in range(traj_radius):
        initial_obs_pool.extend(full_MPC_data['obs'][kp-i:-1:200,0:6].tolist())
        initial_obs_pool.extend(full_MPC_data['obs'][kp+i:-1:200,0:6].tolist())

    if old_obs_pool is None:
        env.customized_reset = initial_obs_pool.copy()
    else:
        samples_from_new = initial_obs_pool[np.random.choice(0.7 * len(initial_obs_pool))]
        samples_from_old = old_obs_pool[np.random.choice(0.3 * len(old_obs_pool))]
        env.customized_reset = samples_from_new.extend(samples_from_old)

    old_obs_pool = initial_obs_pool.copy()
    model.learn(total_timesteps=2e4)


# model.learn(int(1e5))

for _ in range(1000):
        action, _ = model.predict(obs)
        # obs, reward, done, suc, _ = env.step(action)
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        # env.render()
        if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = env.reset()
                # obs = env.reset(to=initial_obs_pool[np.random.choice(len(initial_obs_pool))])

env.close()
# ------------------------------------------------------------------------------