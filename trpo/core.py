import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

# AMEND: added by xlv
def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

# def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
#     for h in hidden_sizes[:-1]:
#         x = tf.layers.dense(x, units=h, activation=activation)
#     return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

# AMEND: added by xlv
def mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None, hidden_initializer=None, output_initializer=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation, kernel_initializer=hidden_initializer)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, kernel_initializer=output_initializer)

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1- mu0)**2 + var0)/(var1 + EPS) - 1) +  log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)

def categorical_kl(logp0, logp1):
    """
    tf symbol for mean KL divergence between two batches of categorical probability distributions,
    where the distributions are input as log probs.
    """
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    return tf.reduce_mean(all_kls)

def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def flat_grad(f, params, clip_norm=None):
    grads = tf.gradients(xs=params, ys=f)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return flat_concat(grads)

def hessian_vector_product(f, params):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    x = tf.placeholder(tf.float32, shape=g.shape)
    return x, flat_grad(tf.reduce_sum(g*x), params)

def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)

    old_logp_all = placeholder(act_dim)
    d_kl = categorical_kl(logp_all, old_logp_all)

    info = {'logp_all': logp_all}
    info_phs = {'logp_all': old_logp_all}

    return pi, logp, logp_pi, info, info_phs, d_kl


def mlp_gaussian_policy(x, a, stochastic, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation, hidden_initializer=normc_initializer(1.0), output_initializer=normc_initializer(0.01))
    log_std = tf.get_variable(name='log_std', initializer=-0.69*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std

    pi = tf.cond(stochastic, lambda: pi, lambda: mu)

    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    old_mu_ph, old_log_std_ph = placeholders(act_dim, act_dim)
    d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)

    info = {'mu': mu, 'log_std': log_std}
    info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

    return pi, logp, logp_pi, info, info_phs, d_kl


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, stochastic, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        policy_outs = policy(x, a, stochastic, hidden_sizes, activation, output_activation, action_space)
        pi, logp, logp_pi, info, info_phs, d_kl = policy_outs
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, info, info_phs, d_kl, v

def mlp_actor_critic_vinit(x, a, stochastic, vinit=None, env=None, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):
    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        policy_outs = policy(x, a, stochastic, hidden_sizes, activation, output_activation, action_space)
        pi, logp, logp_pi, info, info_phs, d_kl = policy_outs
    with tf.variable_scope('v'):

        import os, pickle
        import pandas as pd
        import baselines.logger as logger
        if vinit == 'boltzmann':
            with open(os.environ['PROJ_HOME_3'] + "/tf_model/dubinsCar/vf_weights.pkl", 'rb') as f:
                weights = pickle.load(f)
                print("loading external value weights from value iteration!")

                # --- prepare stats for proper normalization --- #
                print("preparing stats mean and std for normalization ...")
                val_filled_path = os.environ['PROJ_HOME_3'] + "/data/dubinsCar/env_difficult/valFunc_filled_cleaned.csv"
                assert os.path.exists(val_filled_path)
                colnames = ['x', 'y', 'theta', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                val_filled = pd.read_csv(val_filled_path, names=colnames, na_values="?", comment='\t', sep=",",
                                         skipinitialspace=True, skiprows=1)

                stats_source = val_filled.copy()
                stats_source.dropna()
                stats_source.pop('value')
                stats = stats_source.describe()
                stats = stats.transpose()

                x = tf.clip_by_value((x - np.array(stats['mean'])) / np.array(stats['std']), -5.0, 5.0)

                x = tf.layers.dense(inputs=x,
                                    units=64,
                                    name="fc1",
                                    kernel_initializer=tf.constant_initializer(weights[0][0]),
                                    bias_initializer=tf.constant_initializer(weights[0][1]),
                                    use_bias=True,
                                    activation=tf.nn.tanh)

                x = tf.layers.dense(inputs=x,
                                    units=64,
                                    name="fc2",
                                    kernel_initializer=tf.constant_initializer(weights[1][0]),
                                    bias_initializer=tf.constant_initializer(weights[1][1]),
                                    use_bias=True,
                                    activation=tf.nn.tanh)

                v = tf.layers.dense(inputs=x,
                                     units=1,
                                     name="final",
                                     kernel_initializer=tf.constant_initializer(weights[2][0]),
                                     bias_initializer=tf.constant_initializer(weights[2][1]),
                                     use_bias=True)[:, 0]
        elif vinit == 'mpc':
            if env == 'DubinsCarEnv-v0':
                wt_filename = "/tf_model/dubinsCar/vf_mpc_weights.pkl"
                logging = "We are loading external value weights for dubins car trained by mpc cost. this time: no soft constraints data!"
                # logging = "We are loading external value weights for dubins car trained by mpc cost. this time: with soft constraints data!")
                val_filled_path = os.environ['PROJ_HOME_3'] + "/data/dubinsCar/env_difficult/valFunc_mpc_filled_cleaned.csv"
                # val_filled_path = os.environ['PROJ_HOME_3'] + "/data/dubinsCar/env_difficult/valFunc_mpc_filled_cleaned_soft.csv"
                colnames = ['reward', 'value', 'cost', 'status', 'x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5',
                            'd6', 'd7', 'd8']
                # colnames = ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward', 'value', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']

            elif env == 'PlanarQuadEnv-v0':
                wt_filename = "/tf_model/quad/vf_mpc_weights.pkl"
                logging = "We are loading external value weights for quadrotor trained by mpc cost"
                val_filled_path = os.environ['PROJ_HOME_3'] + "/data/quad/valFunc_mpc_filled_cleaned.csv"
                colnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
                            'reward', 'value', 'cost', 'collision_in_future',
                            'collision_current', 'col_trajectory_flag']

            with open(os.environ['PROJ_HOME_3'] + wt_filename, 'rb') as f:
                weights = pickle.load(f)
                print("shape of weights:", np.shape(weights))
                logger.log(logging)

                # --- prepare stats for proper normalization --- #
                print("preparing stats mean and std for normalization ...")
                assert os.path.exists(val_filled_path)
                val_filled = pd.read_csv(val_filled_path, names=colnames, na_values="?", comment='\t', sep=",",
                                         skipinitialspace=True, skiprows=1)

                stats_source = val_filled.copy()
                stats_source.dropna()

                stats = stats_source.describe()
                if env == 'DubinsCarEnv-v0':
                    stats.pop("reward")
                    stats.pop("value")
                    stats.pop("cost")
                    stats.pop("status")
                elif env == 'PlanarQuadEnv-v0':
                    stats.pop("reward")
                    stats.pop("value")
                    stats.pop("cost")
                    stats.pop("collision_in_future")
                    stats.pop("collision_current")
                    stats.pop("col_trajectory_flag")
                else:
                    raise ValueError("gym env is not valid!")
                stats = stats.transpose()

                # obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                x = tf.clip_by_value((x - np.array(stats['mean'])) / np.array(stats['std']), -5.0, 5.0)
                # ----------------------------------------------- #

                out_1 = tf.layers.dense(inputs=x,
                                        units=64,
                                        name="fc1",
                                        kernel_initializer=tf.constant_initializer(weights[0][0]),
                                        bias_initializer=tf.constant_initializer(weights[0][1]),
                                        use_bias=True,
                                        activation=tf.nn.tanh)

                out_2 = tf.layers.dense(inputs=out_1,
                                        units=64,
                                        name="fc2",
                                        kernel_initializer=tf.constant_initializer(weights[1][0]),
                                        bias_initializer=tf.constant_initializer(weights[1][1]),
                                        use_bias=True,
                                        activation=tf.nn.tanh)

                v = tf.layers.dense(inputs=out_2,
                                     units=1,
                                     name="final",
                                     kernel_initializer=tf.constant_initializer(weights[2][0]),
                                     bias_initializer=tf.constant_initializer(weights[2][1]),
                                     use_bias=True)[:, 0]
        else:
            raise ValueError("Please check if your vf_type is valid!")

    return pi, logp, logp_pi, info, info_phs, d_kl, v