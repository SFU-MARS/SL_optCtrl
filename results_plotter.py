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

basedir1 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'difficult_case', 'mpc', 'best_decay_mpc_nn')
basedir2 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'difficult_case', 'baseline')
basedir3 = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests', 'difficult_case', 'vi', 'linear_decay_former_dynamic_deque_latter_nn')

folders1 = [os.path.join(basedir1, d) for d in os.listdir(basedir1)]
folders2 = [os.path.join(basedir2, d) for d in os.listdir(basedir2)]
folders3 = [os.path.join(basedir3, d) for d in os.listdir(basedir3)]
folders = folders1 + folders2 + folders3

rewards = []
iterations = []
hues = []
for subfolder in folders:
    if os.path.isfile(subfolder):
        continue

    fullpath = subfolder
    cur_hue = '_'.join(subfolder.split('/')[-1].split('_')[4:])
    print("hue:", cur_hue)

    max_iter_idx = np.max([int(os.path.splitext(i)[0].split('_')[-1]) for i in os.listdir(os.path.join(fullpath, 'result'))])

    cur_reward = pickle.load(open(os.path.join(fullpath, 'result', 'train_reward_iter_' + str(max_iter_idx) + '.pkl'), 'rb'))

    rewards.extend(cur_reward)
    iterations.extend(range(len(cur_reward)))
    hues.extend([cur_hue]*len(cur_reward))


d = {'hues':hues, 'iterations':iterations, 'rewards':rewards}
df = pd.DataFrame(data=d)
print(df)

ax = sns.lineplot(x="iterations", y="rewards", hue='hues', data=df)

plt.savefig("/home/xlv/Desktop/comparison.png")
plt.show()