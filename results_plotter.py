import seaborn as sns
import sys, os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

case_type = 'difficult_case'
exploration_type = 'linear_decay'

# case_type = 'simple_case'
# exploration_type = 'exponential_decay'

basedir = os.path.join(os.environ['PROJ_HOME_3'], 'runs_log_tests',
                       case_type, exploration_type)
rewards = []
iterations = []
hues = []
for subfolder in os.listdir(basedir):
    fullpath = os.path.join(basedir, subfolder)
    cur_hue = '_'.join(subfolder.split('_')[4:])
    print("hue:", cur_hue)
    cur_reward = pickle.load(open(os.path.join(fullpath, 'result', 'train_reward_iter_150.pkl'), 'rb'))

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
