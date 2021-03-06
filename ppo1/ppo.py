from ppo1.mlp_policy import MlpPolicy_mod
from ppo1.pposgd_simple import *
from ppo1.common import tf_util as U
from collections import Counter
import pickle

import pandas as pd


from utils.tools import *



def create_session(num_cpu=None):
    U.make_session(num_cpu=num_cpu).__enter__()

def create_policy(name, env, args, vf_load=False, pol_load=False):
    ob_space = env.observation_space
    ac_space = env.action_space

    logger.log("i am using mlppolicy_mod")
    logger.log("vf_load is %r, pol_load is %r" % (vf_load, pol_load))
    return MlpPolicy_mod(name=name,ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=2,
                         args=args, load_weights_vf=vf_load, load_weights_pol=pol_load)
    # if vf_load or pol_load:
    #     logger.log("i am using mlppolicy_mod")
    #     return MlpPolicy_mod(name=name,
    #                  ob_space=ob_space, ac_space=ac_space,
    #                  hid_size=64, num_hid_layers=2, load_weights_vf=vf_load, load_weights_pol=pol_load)
    # else:
    #     logger.log("i am using mlppolicy")
    #     return MlpPolicy(name=name,
    #                  ob_space=ob_space, ac_space=ac_space,
    #                  hid_size=64, num_hid_layers=2)

def initialize():
    U.initialize()


def ppo_eval(env, policy, timesteps_per_actorbatch, max_iters, stochastic=False, scatter_collect=False):
    pi = policy
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=stochastic)
    iters_so_far = 0


    episodes_so_far = 0
    timesteps_so_far = 0
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

    return pi


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
    grad_norm = args['grad_norm']
    logger.log("running grad norm:", grad_norm)


    # Setup losses and stuff
    # ----------------------------------------
    pi = policy
    # oldpi = create_policy("oldpi", env) # Network for old policy
    oldpi = create_policy("oldpi", env, args=args, vf_load=True if args['vf_load'] == "yes" else False, pol_load=True if args['pol_load'] == "yes" else False)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    advret = tf.placeholder(dtype=tf.float32, shape=[None])

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    # Uncomment this if you use PPO
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

    # Uncomment this if you use A2C. Remeber to check the optim_stepsize
    # logger.log("trying to use A2C update this time ...")
    # neglogpac = pi.pd.neglogp(ac)
    # pg_loss = tf.reduce_mean(neglogpac * atarg)
    # pol_surr = pg_loss

    # Default we do not use single value NN
    criteron = tf.placeholder(name='criteron', dtype=tf.bool, shape=[])
    cond_val_update = False  # An indicator of value update

    # warning: do not update weights of value network if loading customized external value initialization.
    if args['vf_load'] == "yes":
        if args['vf_switch'] == "yes":
            cond_val_update = True
            vf_loss = tf.cond(criteron, lambda: tf.reduce_mean(tf.square(pi.vpred - ret)),
                              lambda: tf.reduce_mean(tf.square(pi.vpred - pi.vpred)))
            logger.log("loading external valfunc and vf_loss is updated based on MC-TD criteron!")

        elif args['vf_switch'] == "no":
            cond_val_update = False
            vf_loss = tf.reduce_mean(tf.square(pi.vpred - pi.vpred))
            logger.log("loading external valfunc and vf_loss is fixed!")

        elif args['vf_switch'] == 'advanced':
            # not used for now
            vf_loss = tf.reduce_mean(tf.square(pi.vpred - advret))
            logger.log("loading external valfunc and vf_loss is updated based on advanced criteron!")

        else:
            logger.log("the argument for vf_switch is not valid, please double-check it.")

        # XLV: always add one more value NN with non-stop updating
        vf_ghost_loss = tf.reduce_mean(tf.square(pi.vpred_ghost - ret))

    else:
        if args['vf_switch'] == "yes":
            cond_val_update = True
            vf_loss = tf.cond(criteron, lambda: tf.reduce_mean(tf.square(pi.vpred - ret)),
                              lambda: tf.reduce_mean(tf.square(pi.vpred - pi.vpred)))
            logger.log("vf is randomly initialized (ppo baseline), but vf_loss is updated using MC-TD criteron!")

        elif args['vf_switch'] == "no":
            cond_val_update = False
            vf_loss = tf.reduce_mean(tf.square(pi.vpred - pi.vpred))
            logger.log("vf is randomly initialized (ppo baseline), and vf_loss is fixed!")

        elif args['vf_switch'] == "always":
            # this is the default ppo value update strategy. remember to set properly.
            cond_val_update = False
            vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
            logger.log("vf is randomly initialized (ppo baseline), vf_loss is updated every iteration!")


        vf_ghost_loss = tf.reduce_mean(tf.square(pi.vpred_ghost - ret))



    # total_loss = pol_surr + pol_entpen + vf_loss
    # losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    # loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    # XLV: changed for testing value NN switch
    total_loss = pol_surr + pol_entpen + vf_loss + vf_ghost_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent, vf_ghost_loss]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent", "vf_ghost_loss"]

    var_list = pi.get_trainable_variables()
    # print("trainable variables:", var_list)

    pol_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/pol')
    get_pol_surr_grads = U.function([ob, ac, atarg, lrmult], U.flatgrad(pol_surr, pol_var_list))

    if cond_val_update:
        lossandgrad = U.function([ob, ac, atarg, ret, lrmult, criteron], losses + [U.flatgrad(total_loss, var_list)])
    else:
        lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])


    # XLV: added for limit gradient change
    if cond_val_update:
        lossandgrad_clip = U.function([ob, ac, atarg, ret, lrmult, criteron], losses + [U.flatgrad(total_loss, var_list, clip_norm=grad_norm)])
    else:
        lossandgrad_clip = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list, clip_norm=grad_norm)])

    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    if cond_val_update:
        compute_losses = U.function([ob, ac, atarg, ret, lrmult, criteron], losses)
    else:
        compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Initializing oldpi = pi.
    assign_old_eq_new()

    # Prepare for rollouts
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0

    ep_suc_so_far = 0 # success episodes num so far during training
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    ep_mean_rews = list()
    ep_mean_lens = list()

    eval_success_rates = list()   # this is for saving global info for multiple evaluation results.
    eval_suc_buffer = deque(maxlen=2)
    start_clip_grad = False

    # XLV: is it time to switch value NN?
    # switch = False
    val_update_criteron = False

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

        """ Learning rate scheduler """
        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            # cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            cur_lrmult = 1.0
            cur_lrmult = max(cur_lrmult * np.power(0.95, float(iters_so_far) / max_iters), 0.7)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % (iters_so_far + 1)) # Current iteration index

        # seg = seg_gen.__next__()
        # add_vtarg_and_adv(seg, gamma, lam)

        # if seg.pkl exists, read from it. Only used when we do policy gradient analysis
        seg = pickle.load(open("/local-scratch/xlv/SL_optCtrl/theory_analysis_results/seg.pkl", "rb"))

        # one-time use for saving seg data. Comment it
        # pickle.dump(seg, open(args['RUN_DIR'] + "/seg.pkl", "wb"))
        # return 
        
        ob, ac, tdlamret = seg["ob"], seg["ac"], seg["tdlamret"]
        if args['adv_shift'] == "yes":
            logger.log(" --- advantage shift test is enabled --- ")
            tdlamret -= 100
        rews = seg['rew']
        ep_rets = seg['ep_rets']
        event_flags = seg['event_flag']
        train_sucs = seg['suc']
        mc_rets = seg['mcreturn']
        vpredbefore = seg['vpred']

        atarg = seg["adv"] # with default lambda passed from argument
        atarg_ghost = seg['adv_ghost'] # with default lambda passed from argument

        # can be commented if not used any more
        atarg_095 = seg["adv_lam_095"]
        atarg_080 = seg["adv_lam_080"]
        atarg_060 = seg["adv_lam_060"]
        atarg_040 = seg["adv_lam_040"]
        atarg_020 = seg["adv_lam_020"]    

        atarg_ghost_095 = seg["adv_ghost_lam_095"]
        atarg_ghost_080 = seg["adv_ghost_lam_080"]
        atarg_ghost_060 = seg["adv_ghost_lam_060"]
        atarg_ghost_040 = seg["adv_ghost_lam_040"]
        atarg_ghost_020 = seg["adv_ghost_lam_020"]           





        print(len(mc_rets))
        print(len(vpredbefore))
        logger.log("Sum of value pred over this iteration: %f" % np.sum(vpredbefore))
        logger.log("Sum of mc return over this iteration: %f" % np.sum(mc_rets))


        if cond_val_update:
            if np.sum(mc_rets) < np.sum(vpredbefore):
                val_update_criteron = False
                logger.log("We are using MC-TD value update rule. This iter we do not update value function")
            else:
                val_update_criteron = True
                logger.log("We are using MC-TD value update rule. This iter we update value function")

        elif args['vf_switch'] == 'no':
            logger.log("value function is fixed all the time!")
        elif args['vf_switch'] == 'always':
            logger.log("value is always being updated in every iteration!")



        """ In case of collecting real-time sim data and its vpred for furthur debugging """
        if args['vf_load'] == 'no':
            valpred_csv_name = 'ppo_valpred_itself'
        else:
            valpred_csv_name = 'ppo_valpred_external'
        with open(args['RUN_DIR'] + '/' + valpred_csv_name + '.csv', 'a') as f:
            vpred_shaped = vpredbefore.reshape(-1, 1)
            atarg_shaped = atarg.reshape(-1,1)
            tdlamret_shaped = tdlamret.reshape(-1,1)
            rews_shaped = rews.reshape(-1,1)
            event_flags_shaped = np.array(event_flags).reshape(-1,1)

            log_data = np.concatenate((ob, vpred_shaped, atarg_shaped, tdlamret_shaped, rews_shaped, event_flags_shaped), axis=1)

            if args['gym_env'] == 'PlanarQuadEnv-v0':
                log_df = pd.DataFrame(log_data,
                                            columns=['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', valpred_csv_name, 'atarg', 'tdlamret','rews', 'events'])
            elif args['gym_env'] == 'DubinsCarEnv-v0':
                log_df = pd.DataFrame(log_data,
                                            columns=['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', valpred_csv_name, 'atarg', 'tdlamret',  'rews', 'events'])
            elif args['gym_env'] == 'DubinsCarEnv-v1':
                log_df = pd.DataFrame(log_data,
                                            columns=['x', 'y', 'theta', 'v', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', valpred_csv_name, 'atarg', 'tdlamret',  'rews', 'events'])
            else:
                raise ValueError("invalid env !!!")
            log_df.to_csv(f, header=True)


        """ Optimization """
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        atarg_ghost = (atarg_ghost - atarg_ghost.mean()) / atarg_ghost.std() # standardized advantage function estimate for adv_ghost
        
        # comment if not needed
        atarg_095 = (atarg_095 - atarg_095.mean()) / atarg_095.std()
        atarg_080 = (atarg_080 - atarg_080.mean()) / atarg_080.std()
        atarg_060 = (atarg_060 - atarg_060.mean()) / atarg_060.std()
        atarg_040 = (atarg_040 - atarg_040.mean()) / atarg_040.std()
        atarg_020 = (atarg_020 - atarg_020.mean()) / atarg_020.std()  

        atarg_ghost_095 = (atarg_ghost_095 - atarg_ghost_095.mean()) / atarg_ghost_095.std()
        atarg_ghost_080 = (atarg_ghost_080 - atarg_ghost_080.mean()) / atarg_ghost_080.std()
        atarg_ghost_060 = (atarg_ghost_060 - atarg_ghost_060.mean()) / atarg_ghost_060.std()
        atarg_ghost_040 = (atarg_ghost_040 - atarg_ghost_040.mean()) / atarg_ghost_040.std()
        atarg_ghost_020 = (atarg_ghost_020 - atarg_ghost_020.mean()) / atarg_ghost_020.std()


        # d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, atarg_ghost=adv_ghost), shuffle=not pi.recurrent)
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, atarg_ghost=atarg_ghost,
                        atarg_095 = atarg_095, atarg_ghost_095 = atarg_ghost_095,
                        atarg_080 = atarg_080, atarg_ghost_080 = atarg_ghost_080,
                        atarg_060 = atarg_060, atarg_ghost_060 = atarg_ghost_060,
                        atarg_040 = atarg_040, atarg_ghost_040 = atarg_ghost_040,
                        atarg_020 = atarg_020, atarg_ghost_020 = atarg_ghost_020), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

    
        # update pi.ob_rms based on the most recent ob
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values

        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))


        # Here we collect pol_surr gradients with fixed policy network (do not apply adam.update)
        from config import ggl, ggl_ghost  # global ggl, ggl_ghost from config.py
        from config import ggl_095, ggl_080, ggl_060, ggl_040, ggl_020, ggl_ghost_095, ggl_ghost_080,  ggl_ghost_060, ggl_ghost_040, ggl_ghost_020
        logger.log("Start collecting policy gradients for variance analysis ...")
        pga_batchsize = 1
        i = 0
        for batch in d.iterate_once(pga_batchsize):
            print("Processing {}".format(i))
            i = i + 1
            # pol_surr_grads = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg"], cur_lrmult)
            # ggl.append(pol_surr_grads.reshape(-1,1))

            # pol_surr_grads_ghost = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_ghost"], cur_lrmult)
            # ggl_ghost.append(pol_surr_grads_ghost.reshape(-1,1))

            # # comment if not needed
            # # lambda = 0.95
            # pol_surr_grads_095 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_095"], cur_lrmult)
            # ggl_095.append(pol_surr_grads_095.reshape(-1,1))

            # pol_surr_grads_ghost_095 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_ghost_095"], cur_lrmult)
            # ggl_ghost_095.append(pol_surr_grads_ghost_095.reshape(-1,1))

            # # lambda = 0.80
            # pol_surr_grads_080 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_080"], cur_lrmult)
            # ggl_080.append(pol_surr_grads_080.reshape(-1,1))

            # pol_surr_grads_ghost_080 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_ghost_080"], cur_lrmult)
            # ggl_ghost_080.append(pol_surr_grads_ghost_080.reshape(-1,1))

            # # lambda = 0.60
            # pol_surr_grads_060 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_060"], cur_lrmult)
            # ggl_060.append(pol_surr_grads_060.reshape(-1,1))

            # pol_surr_grads_ghost_060 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_ghost_060"], cur_lrmult)
            # ggl_ghost_060.append(pol_surr_grads_ghost_060.reshape(-1,1))

            # # lambda = 0.40
            # pol_surr_grads_040 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_040"], cur_lrmult)
            # ggl_040.append(pol_surr_grads_040.reshape(-1,1))

            # pol_surr_grads_ghost_040 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_ghost_040"], cur_lrmult)
            # ggl_ghost_040.append(pol_surr_grads_ghost_040.reshape(-1,1))

            # lambda = 0.20
            pol_surr_grads_020 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_020"], cur_lrmult)
            ggl_020.append(pol_surr_grads_020.reshape(-1,1))

            pol_surr_grads_ghost_020 = get_pol_surr_grads(batch["ob"], batch["ac"], batch["atarg_ghost_020"], cur_lrmult)
            ggl_ghost_020.append(pol_surr_grads_ghost_020.reshape(-1,1))

        logger.log("End collecting policy gradients ...")
        ggl_basedir = "/local-scratch/xlv/SL_optCtrl/theory_analysis_results"
        # pickle.dump(ggl, open(ggl_basedir + "/ggl.pkl", "wb"))
        # pickle.dump(ggl_ghost, open(ggl_basedir + "/ggl_ghost.pkl", "wb"))
        # pickle.dump(ggl_095, open(ggl_basedir + "/ggl_095.pkl", "wb"))
        # pickle.dump(ggl_ghost_095, open(ggl_basedir + "/ggl_ghost_095.pkl", "wb"))
        # pickle.dump(ggl_080, open(ggl_basedir + "/ggl_080.pkl", "wb"))
        # pickle.dump(ggl_ghost_080, open(ggl_basedir + "/ggl_ghost_080.pkl", "wb"))
        # pickle.dump(ggl_060, open(ggl_basedir + "/ggl_060.pkl", "wb"))
        # pickle.dump(ggl_ghost_060, open(ggl_basedir + "/ggl_ghost_060.pkl", "wb"))
        # pickle.dump(ggl_040, open(ggl_basedir + "/ggl_040.pkl", "wb"))
        # pickle.dump(ggl_ghost_040, open(ggl_basedir + "/ggl_ghost_040.pkl", "wb"))
        pickle.dump(ggl_020, open(ggl_basedir + "/ggl_020.pkl", "wb"))
        pickle.dump(ggl_ghost_020, open(ggl_basedir + "/ggl_ghost_020.pkl", "wb"))
        return pi
        
        # Here we do a bunch of optimization epochs over the data
        start_clip_grad = True  # we also use clip_norm for gradient
        kl_threshold = 0.5  # kl update limit
        for _ in range(optim_epochs):
            losses = []  # list of sublists, each of which gives the loss based on a set of samples with size "optim_batchsize"
            grads = []   # list of sublists, each of which gives the gradients w.r.t all variables based on a set of samples with size "optim_batchsize"
            for batch in d.iterate_once(optim_batchsize):
                if start_clip_grad:
                    if not cond_val_update:
                        *newlosses, g = lossandgrad_clip(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    else:
                        *newlosses, g = lossandgrad_clip(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                         cur_lrmult, val_update_criteron)
                else:
                    if not cond_val_update:
                        *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    else:
                        *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                    cur_lrmult, val_update_criteron)
                if any(np.isnan(g)):
                    logger.log("there are nan in gradients, skip further updating!")
                    break
                if newlosses[3] < kl_threshold:
                    adam.update(g, optim_stepsize * cur_lrmult)
                    # logger.log("KL loss is %f" %(newlosses[3]))
                else:
                    logger.log("KL loss is %f larger than kl_threshold %f, early stop further updating!" % (newlosses[3], kl_threshold))
                    break # break only jump out of the inner loop
                grads.append(g)
                losses.append(newlosses)

            grads_shape = np.array(grads).shape
            grad_norm_checking = np.less_equal(np.array(grads), np.ones(grads_shape) * (grad_norm+0.1))
            if np.all(grad_norm_checking):
                logger.log("gradient norm checking passed! all gradients are clipped to less than {}!".format(grad_norm))
            else:
                logger.log("gradient norm checking failed!")

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            if not cond_val_update:
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            else:
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, val_update_criteron)
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

        ep_mean_lens.append(np.mean(lenbuffer))
        ep_mean_rews.append(np.mean(rewbuffer))

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        logger.record_tabular("EpRewMeanThisIter", np.mean(seg["ep_rets"]))
        logger.record_tabular("EpSuccessThisIter", Counter(train_sucs)[True])
        logger.record_tabular("SucRateThisIter", Counter(train_sucs)[True] / len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        ep_suc_so_far += Counter(train_sucs)[True]
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("EpSuccessSoFar", ep_suc_so_far)
        logger.record_tabular("SucRateSoFar", ep_suc_so_far / episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

        """ Evaluation """
        EVALUATION_FREQUENCY = 10  # 10
        if iters_so_far % EVALUATION_FREQUENCY == 0:

            eval_max_iters = 5
            eval_iters_so_far = 0
            eval_timesteps_per_actorbatch = timesteps_per_actorbatch

            eval_lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
            eval_rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

            eval_episodes_so_far = 0
            eval_timesteps_so_far = 0
            eval_success_episodes_so_far = 0

            # prepare eval episode generator
            eval_seg_gen = traj_segment_generator(pi, env, eval_timesteps_per_actorbatch, stochastic=False)

            logger.log("********** Start evaluating ... ************")
            while True:
                if eval_max_iters and eval_iters_so_far >= eval_max_iters:
                    break

                logger.log("********** Eval Iteration %i ************" % (eval_iters_so_far + 1))

                eval_seg = eval_seg_gen.__next__()

                eval_lrlocal = (eval_seg["ep_lens"], eval_seg["ep_rets"])    # local values
                eval_listoflrpairs = MPI.COMM_WORLD.allgather(eval_lrlocal)  # list of tuples
                eval_lens, eval_rews = map(flatten_lists, zip(*eval_listoflrpairs))
                eval_lenbuffer.extend(eval_lens)
                eval_rewbuffer.extend(eval_rews)
                logger.record_tabular("EpLenMean", np.mean(eval_lenbuffer))
                logger.record_tabular("EpRewMean", np.mean(eval_rewbuffer))
                logger.record_tabular("EpThisIter", len(eval_lens))
                eval_sucs = eval_seg["suc"]
                logger.record_tabular("EpSuccessThisIter", Counter(eval_sucs)[True])


                eval_episodes_so_far += len(eval_lens)
                eval_timesteps_so_far += sum(eval_lens)
                eval_success_episodes_so_far += Counter(eval_sucs)[True]
                logger.record_tabular("EpisodesSoFar", eval_episodes_so_far)
                logger.record_tabular("TimestepsSoFar", eval_timesteps_so_far)
                logger.record_tabular("EpisodesSuccessSoFar", eval_success_episodes_so_far)
                logger.record_tabular("SuccessRateSoFar", eval_success_episodes_so_far * 1.0 / eval_episodes_so_far)

                eval_iters_so_far += 1
                if MPI.COMM_WORLD.Get_rank() == 0:
                    logger.dump_tabular()
            # save success rate from each evaluation into global list
            eval_success_rates.append(eval_success_episodes_so_far * 1.0 / eval_episodes_so_far)
            eval_suc_buffer.append(eval_success_episodes_so_far * 1.0 / eval_episodes_so_far)


        """ Saving model and statistics """
        MODEL_SAVING_FREQ = 30 # 30 is enough for some learning
        if iters_so_far % MODEL_SAVING_FREQ == 0:
            pi.save_model(args['MODEL_DIR'], iteration=iters_so_far)

            # save necessary training statistics
            with open(args['RESULT_DIR'] + '/train_reward_' + 'iter_' + str(iters_so_far) + '.pkl', 'wb') as f_train:
                pickle.dump(ep_mean_rews, f_train)

            # save necessary evaluation statistics
            with open(args['RESULT_DIR'] + '/eval_success_rate_' + 'iter_' + str(iters_so_far) + '.pkl', 'wb') as f_eval:
                pickle.dump(eval_success_rates, f_eval)

        """ Plotting and saving statistics """
        PLOT_FREQUENCY = 10 # 10
        if iters_so_far % PLOT_FREQUENCY == 0:
            # plot training reward performance
            train_plot_x = np.arange(len(ep_mean_rews)) + 1
            train_plot_x = np.insert(train_plot_x, 0, 0)
            train_plot_y = np.insert(ep_mean_rews, 0, ep_mean_rews[0])
            plot_performance(x=train_plot_x, y=train_plot_y, ylabel=r'episode mean reward at each iteration',
                             xlabel='ppo iterations', figfile=os.path.join(args['FIGURE_DIR'], 'train_reward'), title='TRAIN')


            # plot evaluation success rate
            eval_plot_x = (np.arange(len(eval_success_rates)) + 1) * EVALUATION_FREQUENCY
            eval_plot_x = np.insert(eval_plot_x, 0, 0)
            eval_plot_y = np.insert(eval_success_rates, 0, 0)
            plot_performance(x=eval_plot_x, y = eval_plot_y,
                             ylabel=r'eval success rate',
                             xlabel='ppo iterations', figfile=os.path.join(args['FIGURE_DIR'], 'eval_success_rate'),
                             title="EVAL")




    return pi