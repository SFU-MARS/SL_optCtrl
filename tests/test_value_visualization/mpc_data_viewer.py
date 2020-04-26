import os,sys
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

"""
This class is used for visualizing mpc-related value to figure out possible issues

- input
        fn: file name of "valFunc_mpc_filled_cleaned", csv format
- output 
        a figure will be saved at current directory
"""
class viewer(object):
    def __init__(self):
        # self.fn = '/local-scratch/xlv/SL_optCtrl/data/dubinsCar/env_difficult/valFunc_mpc_soft_constraints/valFunc_mpc_filled_cleaned.csv'
        self.fn = '/local-scratch/xlv/SL_optCtrl/data/dubinsCar/env_difficult/valFunc_mpc_filled_cleaned.csv'

    def show_mpc(self, which='value'):
        assert os.path.exists(self.fn)
        assert 'mpc' in self.fn.split('/')[-1].split('_')

        data = pandas.read_csv(self.fn)
        plt_data = data[['x', 'y', which]]

        sns.scatterplot(x='x', y='y', data=plt_data, hue='value')
        plt.savefig("./" + os.path.splitext(self.fn.split('/')[-1])[0] + ".png")
        plt.show()


if __name__ == "__main__":
    vr = viewer()
    vr.show_mpc()