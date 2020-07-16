import seaborn as sns
import sys, os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'good trap env', 'baseline')
basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'good trap env', 'fixed')
basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'good trap env', 'switch')


basedir_list = [basedir1, basedir2, basedir3]


showdir = basedir_list
cues_list=['PPO baseline', 'PPO fixed', 'PPO switch']

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
# choice = 'reward'  # or 'success rate'
# choice = 'success rate'
# choice = 'trap rate'
choice = 'goal rate'

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
    elif choice == "trap rate":
        cur_stats = np.load(os.path.join(fullpath, 'result', 'train_trap_rate_iter_' + str(max_iter_idx) + '.npy'))
    elif choice == "goal rate":
        cur_stats = np.load(os.path.join(fullpath, 'result', 'train_goal_rate_iter_' + str(max_iter_idx) + '.npy'))
    else:
        print("choice is invalid")
    print(cur_stats)
    stats.extend(cur_stats)
    iterations.extend(range(len(cur_stats)))
    hues.extend([cur_hue]*len(cur_stats))


d = {'hues':hues, 'iterations':iterations, choice:stats}
df = pd.DataFrame(data=d)
print(df)

ax = sns.lineplot(x="iterations", y=choice, hue='hues', hue_order=cues_list, data=df)


plt.savefig("/media/anjian/Data/Francis/SL_optCtrl/runs_log_tests/good trap env/train_goal_rate.png")
plt.show()