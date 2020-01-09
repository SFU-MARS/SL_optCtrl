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

case_type = 'difficult_case'
exploration_type = 'linear_decay_nn'

basedir = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',
                       case_type, exploration_type)
rewards = []
iterations = []
hues = []
for subfolder in os.listdir(basedir):
    if os.path.isfile(subfolder):
        continue
    fullpath = os.path.join(basedir, subfolder)
    cur_hue = '_'.join(subfolder.split('_')[4:])
    print("hue:", cur_hue)

    max_iter_idx = np.max([int(os.path.splitext(i)[0].split('_')[-1]) for i in os.listdir(os.path.join(fullpath, 'result'))])

    cur_reward = pickle.load(open(os.path.join(fullpath, 'result', 'train_reward_iter_' + str(max_iter_idx) + '.pkl'), 'rb'))

    rewards.extend(cur_reward)
    iterations.extend(range(len(cur_reward)))
    hues.extend([cur_hue]*len(cur_reward))


d = {'hues':hues, 'iterations':iterations, 'rewards':rewards}
df = pd.DataFrame(data=d)
print(df)

# fmri = sns.load_dataset("fmri")
# print("fmri", fmri)
ax = sns.lineplot(x="iterations", y="rewards", hue='hues', data=df)
plt.show()
