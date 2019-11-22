from ppo1.mlp_policy import MlpPolicy, MlpPolicy_mod
from ppo1.pposgd_simple import *
from collections import defaultdict
from baselines.common import tf_util as U
import numpy as np
from collections import Counter
import globals
import pickle

import pandas as pd
import os, sys

def create_session(num_cpu=None):
    U.make_session(num_cpu=num_cpu).__enter__()


def create_policy(name, env, vf_load=False, pol_load=False):
    ob_space = env.observation_space
    ac_space = env.action_space

    if vf_load or pol_load:
        print("i am using mlppolicy_mod")
        return MlpPolicy_mod(name=name,
                     ob_space=ob_space, ac_space=ac_space,
                     hid_size=64, num_hid_layers=2, load_weights_vf=vf_load, load_weights_pol=pol_load)
    else:
        print("i am using mlppolicy")
        return MlpPolicy(name=name,
                     ob_space=ob_space, ac_space=ac_space,
                     hid_size=64, num_hid_layers=2)

    '''
    print("I'M HERE, NO ANY INITIALIZATION")
    return MlpPolicy(name=name,
                     ob_space=ob_space, ac_space=ac_space,
                     hid_size=64, num_hid_layers=2)
    '''
def initialize():
    U.initialize()


def ppo_eval(env, policy, timesteps_per_actorbatch, max_iters=0, stochastic=False, scatter_collect=False):
    pi = policy
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=stochastic)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    ep_mean_rews = list()
    ep_mean_lens = list()

    # added by xlv
    suc_counter = 0
    ep_counter = 0

    trajs = []
    dones = []
    while True:
        if max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()

        # added by xlv for computing success percentage
        sucs = seg["suc"]
        ep_lens = seg['ep_lens']

        suc_counter += Counter(sucs)[True]
        ep_counter += len(ep_lens)

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        # print("ep_rets:", seg["ep_rets"])
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        # print("reward buffer:", rewbuffer)
        ep_mean_lens.append(np.mean(lenbuffer))
        ep_mean_rews.append(np.mean(rewbuffer))

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("success percentage", suc_counter * 1.0 / ep_counter)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

        if scatter_collect:
            trajs.append(seg['ob'])
            dones.append(seg['new'])

    return pi, ep_mean_lens, ep_mean_rews, suc_counter * 1.0 / ep_counter, trajs, dones

def ppo_learn(env, policy,
        timesteps_per_actorbatch,                       # timesteps per actor per update
        clip_param, entcoeff,                           # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
        gamma, lam,                                     # advantage estimation
        args,
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None,  # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        save_obs=False):
    """This is a direct copy of https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py
    The only reason I copied it here is to update the function to not create a new policy but instead update
    the current one for a few iterations.
    """

    # Setup losses and stuff
    # ----------------------------------------
    pi = policy
    oldpi = create_policy("oldpi", env) # Network for old policy

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

    if args['vf_load'] == "yes":
        vf_loss = tf.reduce_mean(tf.square(pi.vpred - pi.vpred))
        print("loading external valfunc and vf_loss is fixed!")
    else:
        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
        print("not loading external valfunc and vf_loss is updating!")

    # AMEND: do not update weights of value network, if loading customized external value initialization.
    # vf_loss = tf.reduce_mean(tf.square(pi.vpred - pi.vpred))
    # print("vf_loss:", vf_loss)

    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    print("lossandgrad:", lossandgrad)
    # AMEND: added by xlv
    lossandgrad_clip = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list, clip_norm=100.)])

    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Initializing oldpi = pi.
    assign_old_eq_new()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    # rewards_map = defaultdict(list)
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    ep_mean_rews = list()
    ep_mean_lens = list()

    # added by xlv
    suc_counter = 0
    ep_counter  = 0
    start_clip_grad = False

    while True:
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            # cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            cur_lrmult = 1.0
            cur_lrmult = max(cur_lrmult * np.power(0.95, float(iters_so_far) / max_iters), 0.7)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        print("obs shape:", np.shape(ob))



        if save_obs:
            globals.g_iter_id += 1
            tmp_seg = {}
            tmp_seg["ob"] = seg["ob"]
            tmp_seg["new"] = seg["new"]
            with open(globals.g_hm_dirpath + '/iter_' + str(globals.g_iter_id) + '.pkl', 'wb') as f:
                pickle.dump(tmp_seg, f)


        # --- added by xlv for computing success percentage ---
        sucs = seg["suc"]
        ep_lens = seg['ep_lens']
        suc_counter += Counter(sucs)[True]
        ep_counter  += len(ep_lens)
        # -----------------------------------------------------

        vpredbefore = seg["vpred"]  # predicted value function before udpate
        # print("vpred shape:", np.shape(vpredbefore))


        # --- collect real-time sim data and its vpred --- #
        #
        if args['vf_load'] == 'no':
            valpred_csv_name = 'ppo_valpred_itself'
        else:
            valpred_csv_name = 'ppo_valpred_external'

        with open(args['RUN_DIR'] + '/' + valpred_csv_name + '.csv', 'a') as f:
            vpred_shaped = vpredbefore.reshape(-1,1)
            obs_vpred = np.concatenate((ob, vpred_shaped), axis=1)

            if args['gym_env'] == 'PlanarQuadEnv-v0':
                df_obs_vpred = pd.DataFrame(obs_vpred,columns=['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', valpred_csv_name])
            elif args['gym_env'] == 'DubinsCarEnv-v0':
                df_obs_vpred = pd.DataFrame(obs_vpred,
                                            columns=['x', 'y', 'theta',  'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', valpred_csv_name])
            else:
                raise ValueError("invalid env !!!")
            df_obs_vpred.to_csv(f, header=True)
        # ------------------------------------------------ #


        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                if start_clip_grad:
                    *newlosses, g = lossandgrad_clip(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                else:
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                # print("newlosses:", newlosses)
                # print("gradient:", g)
                # print("type:", g.dtype)
                if any(np.isnan(g)):
                    cur_lrmult = cur_lrmult * 0.95
                    start_clip_grad = True
                    continue
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        ep_mean_lens.append(np.mean(lenbuffer))
        ep_mean_rews.append(np.mean(rewbuffer))

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

    # Taking avg of each start's rewards.
    # for start in rewards_map:
    #     rewards_map[start] = np.mean(rewards_map[start])
    # return pi, rewards_map, ep_mean_lens, ep_mean_rews

    # AMEND: added by xlv for success percentage
    logger.record_tabular("success percentage", suc_counter * 1.0 / ep_counter)
    return pi, ep_mean_lens, ep_mean_rews, suc_counter * 1.0 / ep_counter
