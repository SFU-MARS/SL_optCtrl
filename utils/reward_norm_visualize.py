import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    iter = 18

    dir_list = ['./runs/DubinsCarEnv-v0_hand_craft_01-Feb-2019_12-04-57',
                './runs/DubinsCarEnv-v0_ttr_01-Feb-2019_16-49-42']

    reward_pickle_files = [i + '/result/resultppo_reward_iter_' + str(iter) + '.pickle' for i in dir_list]

    fig = plt.figure()

    for idx in range(len(reward_pickle_files)):
        with open(reward_pickle_files[idx], 'rb') as f:
            cur_reward = pkl.load(f)

        np_cur_reward = np.asarray(cur_reward)
        mean_reward = np.mean(np_cur_reward)
        stdev_reward = np.std(np_cur_reward)

        # print(np.shape(np_cur_reward))
        np_cur_reward = (np_cur_reward - mean_reward) / stdev_reward
        plt.plot(np_cur_reward)

    plt.show()