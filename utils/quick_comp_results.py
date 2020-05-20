import os,sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def extract_data_from_log(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)-2):
            if lines[i].startswith('| EpRewMean') and lines[i+2].startswith('| ev_tdlam_before'):
                data.append(float(lines[i].split('|')[2].strip()))

    return data
if __name__ == "__main__":
    comp_file_1 = "/local-scratch/xlv/SL_optCtrl/runs_log/14-Nov-2019_16-59-22PlanarQuadEnv-v0_hand_craft_ppo_vf/log.txt"
    comp_file_2 = "/local-scratch/xlv/SL_optCtrl/runs_log/02-Nov-2019_21-08-22PlanarQuadEnv-v0_hand_craft_ppo/log.txt"
    data_1 = extract_data_from_log(comp_file_1)
    data_2 = extract_data_from_log(comp_file_2)

    iters_1 = list(range(len(data_1)))
    iters_2 = list(range(len(data_2)))
    init_type_1 = ['value'] * len(data_1)
    init_type_2 = ['random'] * len(data_2)
    # print(data_1)
    print(len(data_1))
    print(len(iters_1))

    data = data_1 + data_2
    iters = iters_1 + iters_2
    init_type = init_type_1 + init_type_2

    d = {'init type': init_type, 'iterations': iters, 'reward': data}
    df = pd.DataFrame(data=d)
    # print(df)
    ax = sns.lineplot(x="iterations", y="reward", hue='init type', data=df, ci=None)

    plt.show()