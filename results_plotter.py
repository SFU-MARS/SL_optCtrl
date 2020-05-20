import seaborn as sns
import sys, os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_fixed_value_vi')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_fixed_value_mpc_no_softconstraints')
# basedir4 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_fixed_value_mpc_softconstraints')
# basedir5 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_switch_value_single_valNN')
# basedir6 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_switch_value_double_valNN')
# basedir7 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_switch_value_mpc_single_valNN')
#
# basedir_list = [basedir1, basedir2, basedir3, basedir4, basedir5, basedir6, basedir7]
# showdir = [basedir1, basedir2, basedir3]
# cues_list = ['baseline',
#             'VI & fixed',
#             'MPC & fixed',
#             'MPC with soft constraints & fixed',
#             'VI with single valNN & update',
#             'VI with double valNNs & update',
#             'MPC with single valNN & update']


# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests','quad_task_air_space_202002_Francis_mpcdmin_0.3', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests','quad_task_air_space_202002_Francis_mpcdmin_0.3', 'mpc_fixed')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests','quad_task_air_space_202002_Francis_mpcdmin_0.3', 'mpc_switch')

# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_limited_start_area', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_limited_start_area', 'mpc_fixed')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_limited_start_area', 'mpc_switch')



# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_only_use_feasible', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_only_use_feasible', 'mpc_fixed')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_only_use_feasible', 'mpc_switch')




# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_mpc_truncation', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_mpc_truncation', 'mpc_fixed')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_air_space_202002_Francis_mpc_truncation', 'mpc_switch')
# basedir_list = [basedir1, basedir2, basedir3]

# basedir = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_air_space_202002_Francis_goal_angle_0_60')
# basedir_list = [basedir]

# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_air_space_202002_Francis_mpc_truncation', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_air_space_202002_Francis_old_model_new_reward', 'MPC_switch_new_reward')

# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_mpc_fixed_new_reward')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_mpc_switch_new_reward')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_baseline_new_reward')

# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_tests', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_tests', 'fixed_mpc')

# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_tests_3', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'quad_task_exploration', 'quad_task_tests_3', 'fixed_mpc')



# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_mpc_old_model_new_reward', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_mpc_old_model_new_reward', 'mpc_fixed')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_mpc_old_model_new_reward', 'mpc_switch')


# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_mpc_new_model_new_reward', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_mpc_new_model_new_reward', 'mpc_fixed')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_mpc_new_model_new_reward', 'mpc_switch')

#
# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'mpc_fixed')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'vi_fixed')

# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_baseline_test_adv_shift', '0_60')

# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'baseline')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'mpc_fixed')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'mpc_switch')
# basedir4 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'vi_fixed')
# basedir5 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'vi_gd_fixed')
# basedir6 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'vi_switch')
# basedir7 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'baseline_fixed')
# basedir8 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'baseline_switch')



# basedir_list = [basedir1, basedir2, basedir3, basedir4, basedir5, basedir6, basedir7, basedir8]


basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'baseline')
basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'baseline_fixed')
basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',  'quad_task_exploration', 'quad_task_air_space_202002_Francis_goal_angle_0_60', 'baseline_switch')



basedir_list = [basedir1, basedir2, basedir3]

showdir = basedir_list

cues_list = ['baseline', 'baseline_fixed', 'basline_switch']

# cues_list=['baseline', 'mpc_fixed', 'mpc_switch']
# cues_list = ['baseline']
# cues_list = ['baseline', 'mpc_fixed', 'vi_fixed']
# cues_list = ['baseline', 'baseline_adv_shift']
# cues_list = ['baseline', 'mpc_fixed', 'mpc_switch', 'vi_fixed', 'vi_gd_fixed', 'vi_switch', 'baseline_fixed', 'basline_switch']

folder_cue_dict = {}
for based_dir in basedir_list:
    if based_dir in showdir:
        folders = [os.path.join(based_dir, d) for d in os.listdir(based_dir)]
        cues = [cues_list[basedir_list.index(based_dir)]] * len(folders)
        folder_cue_dict.update(dict(zip(folders, cues)))


stats = []
iterations = []
hues = []

plt.rc('legend',fontsize=8)
choice = 'reward'  # or 'success rate'
#choice = 'success rate'

for k,v in folder_cue_dict.items():
    if os.path.isfile(k):
        continue
    fullpath = k
    cur_hue = v
    print("hue:", cur_hue)

    # max_iter_idx = np.max([int(os.path.splitext(i)[0].split('_')[-1]) for i in os.listdir(os.path.join(fullpath, 'result'))])
    max_iter_idx = np.min([np.max([int(os.path.splitext(i)[0].split('_')[-1]) for i in os.listdir(os.path.join(fullpath, 'result'))]), 300])

    if choice == 'reward':
        cur_stats = pickle.load(open(os.path.join(fullpath, 'result', 'train_reward_iter_' + str(max_iter_idx) + '.pkl'), 'rb'))
    elif choice == 'success rate':
        cur_stats = pickle.load(open(os.path.join(fullpath, 'result', 'eval_success_rate_iter_' + str(max_iter_idx) + '.pkl'), 'rb'))
    else:
        print("choice is invalid")
    stats.extend(cur_stats)
    iterations.extend(range(len(cur_stats)))
    hues.extend([cur_hue]*len(cur_stats))


d = {'hues':hues, 'iterations':iterations, choice:stats}
df = pd.DataFrame(data=d)
print(df)

ax = sns.lineplot(x="iterations", y=choice, hue='hues', hue_order=cues_list, data=df)


# plt.savefig("/home/xlv/Desktop/comparison.png")
plt.show()