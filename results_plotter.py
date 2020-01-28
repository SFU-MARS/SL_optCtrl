import seaborn as sns
import sys, os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# case_type = 'difficult_case'
# exploration_type = 'linear_decay'

# case_type = 'simple_case'
# exploration_type = 'exponential_decay'

# case_type = 'difficult_case'
# exploration_type = 'best_decay_mpc_nn'

# basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'difficult_case', 'mpc', 'best_decay_mpc_nn')
# basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'difficult_case', 'baseline')
# basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'difficult_case', 'vi', 'linear_decay_former_dynamic_deque_latter_nn')

basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_baseline')
basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_fixed_value_vi')
basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_fixed_value_mpc_no_softconstraints')
basedir4 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_fixed_value_mpc_softconstraints')
basedir5 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_switch_value_single_valNN')
basedir6 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_switch_value_double_valNN')
basedir7 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'grad_norm_0.5_kl_0.015_std_0.5_switch_value_mpc_single_valNN')

basedir_list = [basedir1, basedir2, basedir3, basedir4, basedir5, basedir6, basedir7]
cues_list = ['baseline',
            'VI & fixed',
            'MPC with no soft constraints & fixed',
            'MPC with soft constraints & fixed',
            'VI with single valNN & update',
            'VI with double valNNs & update',
            'MPC with single valNN & update']

folder_cue_dict = {}
for based_dir in basedir_list:
    folders = [os.path.join(based_dir, d) for d in os.listdir(based_dir)]
    cues = [cues_list[basedir_list.index(based_dir)]] * len(folders)
    folder_cue_dict.update(dict(zip(folders, cues)))


# folders1 = [os.path.join(basedir1, d) for d in os.listdir(basedir1)]
# folders2 = [os.path.join(basedir2, d) for d in os.listdir(basedir2)]
# folders3 = [os.path.join(basedir3, d) for d in os.listdir(basedir3)]
# folders4 = [os.path.join(basedir4, d) for d in os.listdir(basedir4)]
# folders5 = [os.path.join(basedir5, d) for d in os.listdir(basedir5)]
# folders6 = [os.path.join(basedir6, d) for d in os.listdir(basedir6)]
# folders = folders1 + folders2 + folders3 + folders4 + folders5 + folders6

rewards = []
iterations = []
hues = []

plt.rc('legend',fontsize=8)

for k,v in folder_cue_dict.items():
    if os.path.isfile(k):
        continue
    fullpath = k
    cur_hue = v
    print("hue:", cur_hue)

    # max_iter_idx = np.max([int(os.path.splitext(i)[0].split('_')[-1]) for i in os.listdir(os.path.join(fullpath, 'result'))])
    max_iter_idx = np.min([np.max([int(os.path.splitext(i)[0].split('_')[-1]) for i in os.listdir(os.path.join(fullpath, 'result'))]), 300])
    cur_reward = pickle.load(open(os.path.join(fullpath, 'result', 'train_reward_iter_' + str(max_iter_idx) + '.pkl'), 'rb'))

    rewards.extend(cur_reward)
    iterations.extend(range(len(cur_reward)))
    hues.extend([cur_hue]*len(cur_reward))


d = {'hues':hues, 'iterations':iterations, 'rewards':rewards}
df = pd.DataFrame(data=d)
print(df)

ax = sns.lineplot(x="iterations", y="rewards", hue='hues', data=df)

plt.savefig("/home/xlv/Desktop/comparison_2.png")
plt.show()