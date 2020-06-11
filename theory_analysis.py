import pickle
import numpy as np
import scipy.linalg
import multiprocessing as mp
import time
import seaborn as sns

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

def matrix_norm(sigma):
    start_t = time.time()
    conj_sigma = np.conjugate(sigma)
    # print("conjugate operation tooks:", time.time()-start_t)
    res = np.dot(conj_sigma, sigma)
    # eig_vals, eig_vecs = np.linalg.eig(res)
    eig_vals = scipy.linalg.eigvals(res)
    return np.max(eig_vals)

def f(name, ggl_seg):
    samp_mean = sample_mean(ggl_seg, axis=1)
    sigma = sample_cov(ggl_seg, samp_mean)
    max_eigen_val = matrix_norm(sigma)
    print("finshing one seg ...")
    return {name: max_eigen_val}

# if __name__ == "__main__":
#     ldpath = "/local-scratch/xlv/SL_optCtrl/runs_log_tests/08-Jun-2020_21-34-00DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl.pkl"
#     ggl = pickle.load(open(ldpath, "rb"))

#     print("ggl:", ggl)
#     print("length of ggl:", len(ggl))

#     start_t = time.time()
#     samp_mean = sample_mean(ggl, axis=1)
#     stage1_t = time.time()
#     print("Time on computing mean:", stage1_t - start_t)
#     sigma = sample_cov(ggl, samp_mean)
#     stage2_t = time.time()
#     print("Time on computing sigma:", stage2_t - stage1_t)
#     eig_vals = matrix_norm(sigma)
#     stage3_t = time.time()
#     print("Time on computing max eigen value:", stage3_t - stage2_t)
#     # print("all eig vals:", eig_vals)

# if __name__ == "__main__":
#     num_cores = int(mp.cpu_count())
#     print("local computer has: " + str(num_cores) + "cores")
#     pool = mp.Pool(num_cores)

#     ldpath = "/local-scratch/xlv/SL_optCtrl/runs_log_tests/08-Jun-2020_22-02-29DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/ggl.pkl"
#     ggl = pickle.load(open(ldpath, "rb"))

#     param_dict = {}
#     for i in range(300):
#         param_dict["iter_" + str(i+1)] = ggl[i*160:(i+1)*160]
    
#     results = [pool.apply_async(f, args=(name, param)) for name, param in param_dict.items()]
#     results = [p.get() for p in results]

#     pickle.dump(results, open("./ggl_max_eigens.pkl", "wb"))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    res = pickle.load(open("./ggl_max_eigens.pkl", 'rb'))
    res_dict = {}
    for r in res:
        res_dict.update(r)


    x = list(range(0,300))
    y = []
    for it in x:
        y.append(res_dict['iter_'+str(it+1)])

    ax = sns.scatterplot(x,y)
    plt.show()

