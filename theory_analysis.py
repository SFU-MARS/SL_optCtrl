
import numpy as np
from scipy.linalg import norm
import multiprocessing as mp
import time
import seaborn as sns
import ray

import pickle
import os

global TOTAL
TOTAL = 0

# compute vector mean over a list of column vectors
def sample_mean(samp_list, axis):
    m = np.concatenate(samp_list, axis=axis)
    return np.mean(m, axis=axis).reshape(-1,1)

def sample_cov(samp_list, samp_mean):
    # print("samp_list[0].shape:", samp_list[0].shape)
    # print("samp mean shape:", samp_mean.shape)
    assert samp_list[0].shape == samp_mean.shape
    sigma = np.zeros((samp_mean.shape[0], samp_mean.shape[0]))
    for sp in samp_list:
        sigma += np.dot((sp - samp_mean), (sp - samp_mean).T)
    sigma = sigma / (len(samp_list)-1)
    return sigma

# def matrix_norm(sigma):
#     start_t = time.time()
#     conj_sigma = np.conjugate(sigma)
#     # print("conjugate operation tooks:", time.time()-start_t)
#     res = np.dot(conj_sigma, sigma)
#     # eig_vals, eig_vecs = np.linalg.eig(res)
#     eig_vals = scipy.linalg.eigvals(res)
#     return np.max(eig_vals)

def matrix_norm(sigma):
    return norm(sigma, 2)


@ray.remote
def f(name, ggl_seg):
    samp_mean = sample_mean(ggl_seg, axis=1)
    sigma = sample_cov(ggl_seg, samp_mean)
    max_eigen_val = matrix_norm(sigma)
    global TOTAL
    TOTAL += 1
    print("finshing {} seg ...".format(TOTAL))
    return {name: max_eigen_val}


class pgvar_analyser(object):
    def __init__(self, pg_loadpath, res_savepath, iters, bunch_size):
        self.num_cores = int(mp.cpu_count())
        self.pg_loadpath = pg_loadpath
        self.res_savepath = res_savepath
        self.iters = iters            # how many iterations to show on X-axis, normally 300
        self.bunch_size = bunch_size  # sizeof(ggl) / iters

        print("local computer has: " + str(self.num_cores) + "cores")
        ray.init(num_cpus=self.num_cores, ignore_reinit_error=True)

    def run(self):
        assert os.path.exists(self.pg_loadpath)
        ggl = pickle.load(open(self.pg_loadpath, "rb"))
        ggl_length = len(ggl)
        print("length of ggl:", len(ggl))
        assert self.iters * self.bunch_size == ggl_length

        param_dict = {}
        for i in range(self.iters):
            param_dict["iter_" + str(i+1)] = ggl[i*self.bunch_size:(i+1)*self.bunch_size]
        
        results = ray.get([f.remote(name, param) for name, param in param_dict.items()])
        pickle.dump(results, open(self.res_savepath, "wb"))

def Eucliean(x1, x2):
    return np.linalg.norm(x1 - x2)

def Manhattan(x1, x2):
    return np.linalg.norm(x1 - x2, ord=1)

def Cosine(x1, x2):
    return np.dot(x1.T, x2)/(np.linalg.norm(x1)*(np.linalg.norm(x2)))

def Pearson(x1, x2):
    pass



# pg_true =  pickle.load(open("/local-scratch/xlv/SL_optCtrl/runs_log_tests/29-Jun-2020_15-52-52DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl_true.pkl", "rb"))
# pg_bsl_approx = pickle.load(open("/local-scratch/xlv/SL_optCtrl/runs_log_tests/29-Jun-2020_15-52-52DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl_ghost.pkl", "rb"))
# pg_vinit_approx = pickle.load(open("/local-scratch/xlv/SL_optCtrl/runs_log_tests/29-Jun-2020_15-52-52DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl.pkl", "rb"))

# mean_pg_true = np.mean(pg_true, axis=0)
# print("mean pg true:", mean_pg_true.shape)

# batch_options = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 102400]
# res = []
# for i, batch_i in enumerate(batch_options):

#     mean_pg_bsl_approx = np.mean(pg_bsl_approx[:batch_i], axis=0)
#     mean_pg_vinit_approx = np.mean(pg_vinit_approx[:batch_i], axis=0)


#     print("... Using Euclidean Distance ...")
#     Euclidean_bsl = Eucliean(mean_pg_true, mean_pg_bsl_approx)
#     Euclidean_vinit = Eucliean(mean_pg_true, mean_pg_vinit_approx)
#     print("bsl: {}".format(Euclidean_bsl))
#     print("vinit: {}".format(Euclidean_vinit))

#     print("... Using Manhattan Distance ...")
#     Manhattan_bsl = Manhattan(mean_pg_true, mean_pg_bsl_approx)
#     Manhattan_vinit = Manhattan(mean_pg_true, mean_pg_vinit_approx)
#     print("bsl: {}".format(Manhattan_bsl))
#     print("vinit: {}".format(Manhattan_vinit))

#     print("... Using Cosine ...")
#     Cosine_bsl = Cosine(mean_pg_true, mean_pg_bsl_approx)
#     Cosine_vinit = Cosine(mean_pg_true, mean_pg_vinit_approx)
#     print("bsl: {}".format(Cosine_bsl))
#     print("vinit: {}".format(Cosine_vinit))

#     res.append({"Euclidean_bsl":Euclidean_bsl, "Euclidean_vinit":Euclidean_vinit, "Manhattan_bsl":Manhattan_bsl,"Manhattan_vinit":Manhattan_vinit,  "Cosine_bsl":Cosine_bsl, "Cosine_vinit":Cosine_vinit})
# pickle.dump(res, open("res.pkl", "wb"))


import matplotlib.pyplot as plt
res = pickle.load(open("./res.pkl", "rb"))
batch_options = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 102400]
x = []
y = []
hue = []
for i, b in enumerate(batch_options):
    x.append(b)
    y.append(res[i]["Cosine_bsl"][0][0])
    hue.append("Cosine_bsl")

    x.append(b)
    y.append(res[i]["Cosine_vinit"][0][0])
    hue.append("Cosine_vinit")

# ax = sns.scatterplot(x, y, hue)
ax = sns.barplot(x,y,hue)
plt.show()

# if __name__ == "__main__":
    # pgv = pgvar_analyser(pg_loadpath="/local-scratch/xlv/SL_optCtrl/runs_log_tests/22-Jun-2020_15-10-53DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl.pkl",
    #                 res_savepath="./final2_ggl_max_eigens_vinit_lam_1.pkl", 
    #                 iters=300, 
    #                 bunch_size=1024)
    # pgv.run()


    # pgv = pgvar_analyser(pg_loadpath="/local-scratch/xlv/SL_optCtrl/runs_log_tests/22-Jun-2020_15-10-53DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl_ghost.pkl",
    #                 res_savepath="./final2_ggl_max_eigens_baseline.pkl", 
    #                 iters=300, 
    #                 bunch_size=1024)
    # pgv.run()



# Draw the pg var along training iterations using seaborn scatterplot
# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     res_vinit_MC = pickle.load(open("./final2_ggl_max_eigens_vinit_lam_1.pkl", 'rb'))
#     res_bsl = pickle.load(open("./final2_ggl_max_eigens_baseline.pkl", 'rb'))

#     res_vinit_MC_dict = {}
#     for r in res_vinit_MC:
#         res_vinit_MC_dict.update(r)

#     res_bsl_dict = {}
#     for r in res_bsl:
#         res_bsl_dict.update(r)

#     ref = list(range(0,300))

#     x = []
#     y = []
#     hue = []

#     for id, it in enumerate(ref):
#         if res_vinit_MC_dict['iter_'+str(id+1)] < 10000:
#             x.append(id)
#             y.append(res_vinit_MC_dict['iter_'+str(id+1)])
#             hue.append("vinit_MC")
    
#     for id, it in enumerate(ref):
#         if res_bsl_dict['iter_'+str(id+1)] < 10000:
#             x.append(id)
#             y.append(res_bsl_dict['iter_'+str(id+1)])
#             hue.append("bsl")

#     ax = sns.scatterplot(x, y, hue)
#     plt.show()



# Draw the gradient magnitude using seaborn scatterplot
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     ldpath_vinit_TD = "/local-scratch/xlv/SL_optCtrl/runs_log_tests/08-Jun-2020_22-02-29DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl.pkl"
#     ldpath_bsl = "/local-scratch/xlv/SL_optCtrl/runs_log_tests/11-Jun-2020_22-01-21DubinsCarEnv-v0_hand_craft_ppo/ggl.pkl"
    
#     ldpath_vinit_MC = "/local-scratch/xlv/SL_optCtrl/runs_log_tests/15-Jun-2020_22-25-51DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl.pkl"
    
    
#     ggl_vinit_MC = pickle.load(open(ldpath_vinit_MC, "rb"))
#     ggl_vinit_TD = pickle.load(open(ldpath_vinit_TD, "rb"))
#     ggl_bsl = pickle.load(open(ldpath_bsl, "rb"))

#     normed_ggl_vinit_MC = [matrix_norm(sigma) for sigma in ggl_vinit_MC]
#     normed_ggl_vinit_TD = [matrix_norm(sigma) for sigma in ggl_vinit_TD]
#     normed_ggl_bsl = [matrix_norm(sigma) for sigma in ggl_bsl]

#     rand_ggl_vinit_MC = np.random.choice(normed_ggl_vinit_MC, size=1000, replace=False)
#     rand_ggl_vinit_TD = np.random.choice(normed_ggl_vinit_TD, size=1000, replace=False)
#     rand_ggl_bsl = np.random.choice(normed_ggl_bsl, size=1000, replace=False)
    
#     x = list(range(0,len(rand_ggl_bsl)))
#     y = []
#     hue = []
    
#     y.extend(rand_ggl_bsl)
#     hue.extend(['bsl'] * len(rand_ggl_bsl))
#     y.extend(rand_ggl_vinit_TD)
#     hue.extend(['vinit_TD'] * len(rand_ggl_vinit_TD))
#     y.extend(rand_ggl_vinit_MC)
#     hue.extend(['vinit_MC'] * len(rand_ggl_vinit_MC))


#     ax = sns.scatterplot(x+x+x, y, hue, s=30)
#     plt.show()
