import matplotlib.pyplot as plt
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
    assert samp_list[0].shape == samp_mean.shape
    sigma = np.zeros((samp_mean.shape[0], samp_mean.shape[0]))
    for sp in samp_list:
        sigma += np.dot((sp - samp_mean), (sp - samp_mean).T)
    sigma = sigma / (len(samp_list)-1)
    return sigma

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

# This is for analysing the pg varaince
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

iterations = ["0", "20", "40", "60", "80"]

for iteration in iterations:
    workspace = "/media/anjian/Data/Francis/SL_optCtrl/runs_log_tests/experiments_for_calculate_gradient/segments/exp3/iter_" + iteration + "/"
    print(workspace)

    pg_bsl_all   = pickle.load(open(workspace + "ggl_ghost_095.pkl", "rb"))
    pg_vinit_all = pickle.load(open(workspace + "ggl_095.pkl", "rb"))

    pg_bsl_all = np.array(pg_bsl_all)
    pg_vinit_all = np.array(pg_vinit_all)


    pg_bsl_true = np.mean(pg_bsl_all, axis=0)  # true pg using bsl method
    pg_vinit_true = np.mean(pg_vinit_all, axis=0)  # true pg using vinit method

    # batch_options = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    # batch_options = [128 // 64, 256 // 64, 512 // 64, 1024 // 64, 2048 // 64, 4096 // 64, 8192 // 64, 16384 // 64, 32768 // 64]#, 65536, 102400]
    times_options = [100, 100, 100, 100, 100, 100, 100,  100,  100, 1, 1]
    # times_options = [10, 10, 10, 10, 10, 10, 10,  10,  10]#, 1, 1, 1, 1, 1]
    # times_options = [5, 5, 5, 5, 5, 5, 5, 5, 5]#, 5, 5, 5, 5, 5]
    # times_options = [1, 1, 1, 1, 1, 1, 1, 1, 1]#, 1, 1, 1, 1, 1]


                    #128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 102400
    batch_options = [2,   4,   8,   16,   32,   64,   128,  256,   512,   1024,  1600]

    # print(batch_options)

    res = []

    cosine_x = []
    cosine_y = []
    cosine_hue = []

    Eucliean_x = []
    Eucliean_y = []
    Eucliean_hue = []

    index = np.arange(pg_bsl_all.shape[0])

    for i, batch_i in enumerate(batch_options):


        index_list = np.random.randint(1600 - 512, size=times_options[i])


        Euclidean_bsl_bsl_list = []
        Euclidean_vinit_vinit_list = []
        Euclidean_bsl_vinit_list = []
        Euclidean_vinit_bsl_list = []

        Cosine_bsl_bsl_list = []
        Cosine_vinit_vinit_list = []
        Cosine_bsl_vinit_list = []
        Cosine_vinit_bsl_list = []


        for index in index_list:

            if (batch_i > 512):
                index = 0


            pg_bsl_approx   = np.mean(pg_bsl_all[index : index + batch_i], axis=0)
            pg_vinit_approx = np.mean(pg_vinit_all[index : index + batch_i], axis=0)

            # print("... Using Euclidean Distance ...")
            Euclidean_bsl_bsl = Eucliean(pg_bsl_true, pg_bsl_approx)
            Euclidean_vinit_vinit = Eucliean(pg_vinit_true, pg_vinit_approx)
            Euclidean_bsl_vinit = Eucliean(pg_vinit_true, pg_bsl_approx)
            Euclidean_vinit_bsl = Eucliean(pg_bsl_true, pg_vinit_approx)

            Euclidean_bsl_bsl_list.append(Euclidean_bsl_bsl)
            Euclidean_vinit_vinit_list.append(Euclidean_vinit_vinit)
            Euclidean_bsl_vinit_list.append(Euclidean_bsl_vinit)
            Euclidean_vinit_bsl_list.append(Euclidean_vinit_bsl)

            # print("... Using Manhattan Distance ...")
            # Manhattan_bsl_bsl = Manhattan(pg_bsl_true, pg_bsl_approx)
            # Manhattan_vinit_vinit = Manhattan(pg_vinit_true, pg_vinit_approx)
            # Manhattan_bsl_vinit = Manhattan(pg_vinit_true, pg_bsl_approx)
            # Manhattan_vinit_bsl = Manhattan(pg_bsl_true, pg_vinit_approx)

            # print("... Using Cosine ...")
            Cosine_bsl_bsl = Cosine(pg_bsl_true, pg_bsl_approx)
            Cosine_vinit_vinit = Cosine(pg_vinit_true, pg_vinit_approx)
            Cosine_bsl_vinit = Cosine(pg_vinit_true, pg_bsl_approx)
            Cosine_vinit_bsl = Cosine(pg_bsl_true, pg_vinit_approx)
            
            Cosine_bsl_bsl_list.append(Cosine_bsl_bsl[0][0])
            Cosine_vinit_vinit_list.append(Cosine_vinit_vinit[0][0])
            Cosine_bsl_vinit_list.append(Cosine_bsl_vinit[0][0])
            Cosine_vinit_bsl_list.append(Cosine_vinit_bsl[0][0])


        res.append({"Euclidean_bsl_bsl":Euclidean_bsl_bsl, 
                    "Euclidean_vinit_vinit":Euclidean_vinit_vinit, 
                    "Euclidean_bsl_vinit":Euclidean_bsl_vinit,
                    "Euclidean_vinit_bsl":Euclidean_vinit_bsl,  
                    # "Manhattan_bsl_bsl":Manhattan_bsl_bsl, 
                    # "Manhattan_vinit_vinit":Manhattan_vinit_vinit,
                    # "Manhattan_bsl_vinit": Manhattan_bsl_vinit,
                    # "Manhattan_vinit_bsl":Manhattan_vinit_bsl,
                    "Cosine_bsl_bsl":Cosine_bsl_bsl,
                    "Cosine_vinit_vinit":Cosine_vinit_vinit,
                    "Cosine_bsl_vinit":Cosine_bsl_vinit,
                    "Cosine_vinit_bsl":Cosine_vinit_bsl})


        Cosine_bsl_bsl = np.mean(Cosine_bsl_bsl_list)
        Cosine_vinit_vinit = np.mean(Cosine_vinit_vinit_list)
        Cosine_bsl_vinit = np.mean(Cosine_bsl_vinit_list)
        Cosine_vinit_bsl = np.mean(Cosine_vinit_bsl_list)


        Euclidean_bsl_bsl = np.mean(Euclidean_bsl_bsl_list)
        Euclidean_vinit_vinit = np.mean(Euclidean_vinit_vinit_list)
        Euclidean_bsl_vinit = np.mean(Euclidean_bsl_vinit_list)
        Euclidean_vinit_bsl = np.mean(Euclidean_vinit_bsl_list)


        log_batch_i = np.log2(batch_i)

        cosine_x.append(log_batch_i)
        cosine_y.append(Cosine_bsl_vinit)
        cosine_hue.append("Cosine_bsl_vinit")

        cosine_x.append(log_batch_i)
        cosine_y.append(Cosine_vinit_vinit)
        cosine_hue.append("Cosine_vinit_vinit")

        cosine_x.append(log_batch_i)
        cosine_y.append(Cosine_bsl_bsl)
        cosine_hue.append("Cosine_bsl_bsl")

        cosine_x.append(log_batch_i)
        cosine_y.append(Cosine_vinit_bsl)
        cosine_hue.append("Cosine_vinit_bsl")


        Eucliean_x.append(log_batch_i)
        Eucliean_y.append(Euclidean_bsl_vinit)
        Eucliean_hue.append("Euclidean_bsl_vinit")

        Eucliean_x.append(log_batch_i)
        Eucliean_y.append(Euclidean_vinit_vinit)
        Eucliean_hue.append("Euclidean_vinit_vinit")

        Eucliean_x.append(log_batch_i)
        Eucliean_y.append(Euclidean_bsl_bsl)
        Eucliean_hue.append("Euclidean_bsl_bsl")

        Eucliean_x.append(log_batch_i)
        Eucliean_y.append(Euclidean_vinit_bsl)
        Eucliean_hue.append("Euclidean_vinit_bsl")

    # y-axis Cosine similarity
    # sample size (2^x)


    plt.rc('legend',fontsize=25)
    plt.rc('axes', titlesize=30)
    plt.rc('axes', labelsize=30)  

    import matplotlib.pyplot
    ax = matplotlib.pyplot.figure(figsize=(12.0, 8.0))

    ax = sns.scatterplot(cosine_x,cosine_y,cosine_hue, palette=['blue', 'red', 'blue', 'red'], legend = False,
                         hue_order = ["Cosine_vinit_bsl", "Cosine_bsl_bsl", "Cosine_vinit_vinit", "Cosine_bsl_vinit"])
    ax.set_xticks(range(len(batch_options)))
    ax.set_xticklabels([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    ax = sns.lineplot(cosine_x, 
                 cosine_y,
                 cosine_hue,
                 palette=['blue', 'red', 'blue', 'red'],
                 hue_order = ["Cosine_vinit_bsl", "Cosine_bsl_bsl", "Cosine_vinit_vinit", "Cosine_bsl_vinit"])

    ax.lines[2].set_linestyle("--")
    ax.lines[3].set_linestyle("--")

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(30)

    
    ax.set(xlabel='Sample size $2^{x}$', ylabel='Cosine similarity')  

    from matplotlib.lines import Line2D
    lines = [Line2D([0], [0], color="blue", linewidth=3),
             Line2D([0], [0], color="red", linewidth=3),
             Line2D([0], [0], color="blue", linewidth=3, linestyle='--'),
             Line2D([0], [0], color="red", linewidth=3, linestyle='--')]
    labels = ['VI-fixed EST v.s. Benchmark GT', 
              'Benchmark EST v.s. Benchmark GT', 
              'VI-fixed EST v.s. VI-fixed GT', 
              'Benchmark EST v.s. VI-fixed GT']

    plt.legend(lines, labels)



    # plt.show()
    plt.savefig(workspace + 'cosine_distance.pdf')
    plt.clf()




    ax = sns.scatterplot(Eucliean_x, Eucliean_y, Eucliean_hue, 
                         palette=['blue', 'red', 'blue', 'red'], legend = False,
                         hue_order = ["Euclidean_vinit_bsl", "Euclidean_bsl_bsl", "Euclidean_vinit_vinit", "Euclidean_bsl_vinit"])

    ax.set_xticks(range(len(batch_options)))

    ax.set_xticklabels([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    ax = sns.lineplot(Eucliean_x, Eucliean_y, Eucliean_hue, 
                      palette=['blue', 'red', 'blue', 'red'],
                      hue_order = ["Euclidean_vinit_bsl", "Euclidean_bsl_bsl", "Euclidean_vinit_vinit", "Euclidean_bsl_vinit"])

    ax.lines[2].set_linestyle("--")
    ax.lines[3].set_linestyle("--")

    ax.set(xlabel='Sample size', ylabel='Euclidean distance')    

    lines = [Line2D([0], [0], color="blue", linewidth=3),
             Line2D([0], [0], color="red", linewidth=3),
             Line2D([0], [0], color="blue", linewidth=3, linestyle='--'),
             Line2D([0], [0], color="red", linewidth=3, linestyle='--')]
    labels = ['VI-fixed EST v.s. Benchmark GT', 
              'Benchmark EST v.s. Benchmark GT', 
              'VI-fixed EST v.s. VI-fixed GT', 
              'Benchmark EST v.s. VI-fixed GT']

    plt.legend(lines, labels)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(30)


    # plt.show()
    plt.savefig(workspace + 'eucliean_distance.pdf')
    plt.clf()


#blue dash line -> -> baseline trap
#blue solid line -> vi trap
#red dash line -> baseline goal
#red solid line -> vi goal




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
# import matplotlib.pyplot as plt
# res = pickle.load(open("./res.pkl", "rb"))
# # batch_options = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 102400]
# batch_options = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# x = []
# y = []
# hue = []
# for i, b in enumerate(batch_options):
#     x.append(np.log(b))
#     y.append(res[i]["Cosine_bsl"][0][0])
#     hue.append("Cosine_bsl")

#     x.append(np.log(b))
#     y.append(res[i]["Cosine_vinit"][0][0])
#     hue.append("Cosine_vinit")

# sns.scatterplot(x, y, hue)
# sns.lineplot(x,y,hue)
# plt.show()

# if __name__ == "__main__":
#     workspace = "/media/anjian/Data/Francis/SL_optCtrl/runs_log_tests/18-Jul-2020_22-10-27PlanarQuadEnv-v0_hand_craft_ppo_vf_boltzmann/"
#     pgv = pgvar_analyser(pg_loadpath=workspace + "ggl_100.pkl",
#                     res_savepath=workspace + "variance_init.pkl", 
#                     iters=150, 
#                     bunch_size=1024)
#     pgv.run()


#     pgv = pgvar_analyser(pg_loadpath=workspace + "ggl_ghost_100.pkl",
#                     res_savepath=workspace + "variance_baseline.pkl", 
#                     iters=150, 
#                     bunch_size=1024)
#     pgv.run()

#     workspace = "/media/anjian/Data/Francis/SL_optCtrl/runs_log_tests/19-Jul-2020_02-18-09PlanarQuadEnv-v0_hand_craft_ppo/"
#     pgv = pgvar_analyser(pg_loadpath=workspace + "ggl_100.pkl",
#                     res_savepath=workspace + "variance_init.pkl", 
#                     iters=150, 
#                     bunch_size=1024)
#     pgv.run()


#     pgv = pgvar_analyser(pg_loadpath=workspace + "ggl_ghost_100.pkl",
#                     res_savepath=workspace + "variance_baseline.pkl", 
#                     iters=150, 
#                     bunch_size=1024)
#     pgv.run()



    # Boxplots for baseline and initialization, for 2 experiments.

    #############################################################


# Draw the pg var along training iterations using seaborn scatterplot
# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     res_vinit_MC = pickle.load(open("./theory_analysis_results/variance analysis/final_ggl_max_eigens_vinit_lam_1", 'rb'))
#     res_bsl = pickle.load(open("./theory_analysis_results/variance analysis/final_ggl_max_eigens_baseline", 'rb'))

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
#         if res_vinit_MC_dict['iter_'+str(id+1)] < 2000:
#             x.append(id)
#             y.append(res_vinit_MC_dict['iter_'+str(id+1)])
#             hue.append("vinit_MC")
    
#     for id, it in enumerate(ref):
#         if res_bsl_dict['iter_'+str(id+1)] < 2000:
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
