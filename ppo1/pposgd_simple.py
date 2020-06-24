from ppo1.common.dataset import Dataset
from ppo1.common import explained_variance, fmt_row, zipsame
from utils import logger
import ppo1.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from ppo1.common.mpi_adam import MpiAdam
from ppo1.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque


def traj_segment_generator(pi, env, horizon, stochastic, difficulty):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    # BI: Added these for our work.
    # start, ob = env.reset(ret_state_and_ob=True)
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # BI: Added these for our work.
    # start_rews keeps track of the reward from each start.
    # start_rews = defaultdict(list)

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    # XLV: add success percentage
    suc = False
    sucs = np.zeros(horizon, 'int32')
    # XLV: add event flag
    event_flag = 'safe'
    event_flags = [None] * horizon
    # XLV: add vpred_ghost
    vpreds_ghost = np.zeros(horizon, 'float32')

    if (diffculty == "trap"):
        prior = False
        multiple_goal = True
    else:
        prior = True
        multiple_goal = False


    while True:
        prevac = ac
        # ac, vpred = pi.act(stochastic, ob)
        ac, vpred, vpred_ghost = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:

            # AMEND: added by xlv for reset
            ob = env.reset()
            # AMEND: add by xlv for deal with 'nan' data
            if len(ep_rets) == 0 and len(ep_lens) == 0:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0

            # yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new": news, "suc": sucs, "event_flag":event_flags,
            #         "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
            #         "ep_rets" : ep_rets, "ep_lens" : ep_lens}

            yield {"ob": obs, "rew": rews, "vpred": vpreds, "vpred_ghost":vpreds_ghost, "new": news, "suc": sucs, "event_flag": event_flags,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new), "nextvpred_ghost": vpred_ghost * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}

                    # BI: Added this for our work.
                    #"start_rews": start_rews}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        # XLV: added for collect success rate
        sucs[i] = suc
        # XLV: added for collect event type
        event_flags[i] = event_flag
        # XLV: added for maintain vpred_ghost
        vpreds_ghost[i] = vpred_ghost

        ob, rew, new, info = env.step(ac, prior, multiple_goal)
        suc = info['suc']
        event_flag = info['event']
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            cur_ep_ret = 0
            cur_ep_len = 0

            ob = env.reset()

        t += 1

def add_vtarg_and_adv(seg, gamma, lam, method):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    # AMEND: added by xlv for computing Mento-Carlo Return
    G = np.append(seg["rew"], 0)
    tdtarget = np.empty(T, 'float32')
    ######## PPO Version ########################################
    if (method == 'ppo'):
        print(method)
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            # AMEND: added by xlv, update each timestep of G and tdtarget
            G[t] = rew[t] + gamma * G[t+1] * nonterminal
            tdtarget[t] = rew[t] + gamma * vpred[t+1] * nonterminal
        seg["tdlamret"] = seg["adv"] + seg["vpred"]
        # AMEND: added by xlv, return G except the last element, and tdtarget
        seg["mcreturn"] = G[:-1]
        seg["tdtarget"] = tdtarget
    #############################################################

    ######## A2C Version ########################################
    elif (method == 'a2c'):
        print(method)
        tdtarget = np.append(tdtarget, 0)
        for t in reversed(range(T)):
            nonterminal = 1 - new[t+1]
            tdtarget[t] = rew[t] + gamma * tdtarget[t+1] * nonterminal
            gaelam[t] = lastgaelam = tdtarget[t] - vpred[t]
            G[t] = gaelam[t]
        tdtarget = tdtarget[:-1]

        seg["tdtarget"] = tdtarget
        seg["mcreturn"] = G[:-1]
        seg["tdlamret"] = tdtarget
    ##############################################################

def add_vtarg_and_adv_ghost(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred_ghost = np.append(seg["vpred_ghost"], seg["nextvpred_ghost"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred_ghost[t+1] * nonterminal - vpred_ghost[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred_ghost"]

def learn(env, policy_fn,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
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
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
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
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
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
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
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
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
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

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
