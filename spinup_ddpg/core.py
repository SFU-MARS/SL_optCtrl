import numpy as np
import scipy.signal

import torch
import torch.nn as nn

import pickle




def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        cur_layer = nn.Linear(sizes[j], sizes[j+1])
        if j == len(sizes)-2:
            nn.init.uniform_(cur_layer.weight, a=-0.003, b=0.003)
        layers += [cur_layer, act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])



# class MLPActor(nn.Module):
#
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
#         super().__init__()
#         pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
#         self.pi = mlp(pi_sizes, activation, nn.Tanh)
#         self.act_limit = act_limit
#
#     def forward(self, obs):
#         # Return output from network scaled to action space limits.
#         # print("obs:", obs)
#         # print("original action output:", self.pi(obs))
#         # print("multiply act limit:", self.act_limit * self.pi(obs))
#         return self.act_limit * self.pi(obs)





# class MLPQFunction(nn.Module):
#
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
#         super().__init__()
#         self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
#
#     def forward(self, obs, act):
#         q = self.q(torch.cat([obs, act], dim=-1))
#         return torch.squeeze(q, -1) # Critical to ensure q has right shape.



class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, initQ=False):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

        self.normmean = None
        self.normstd = None
        if initQ:
            self.normmean = torch.tensor([-0.051383,
                                          0.028844,
                                          0.002441,
                                          3.532747,
                                          3.533138,
                                          3.529948,
                                          3.521403,
                                          3.524155,
                                          3.522325,
                                          3.521256,
                                          3.526331])
            self.normstd = torch.tensor([2.793068,
                                         2.803955,
                                         1.815369,
                                         2.586505,
                                         2.591617,
                                         2.591861,
                                         2.587785,
                                         2.590880,
                                         2.587919,
                                         2.584125,
                                         2.589151])

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        # print("obs:", obs)
        # print("original action output:", self.pi(obs))
        # print("multiply act limit:", self.act_limit * self.pi(obs))
        if self.normmean is not None:
            obs = (obs - self.normmean) / self.normstd
            return self.act_limit * self.pi(obs)
        else:
            return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, initQ=False):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

        self.normmean = None
        self.normstd  = None
        if initQ:
            self.normmean = torch.tensor([-0.051383,
                                          0.028844,
                                          0.002441,
                                          3.532747,
                                          3.533138,
                                          3.529948,
                                          3.521403,
                                          3.524155,
                                          3.522325,
                                          3.521256,
                                          3.526331,
                                          0.001888,
                                          0.000586])
            self.normstd = torch.tensor([2.793068,
                                         2.803955,
                                         1.815369,
                                         2.586505,
                                         2.591617,
                                         2.591861,
                                         2.587785,
                                         2.590880,
                                         2.587919,
                                         2.584125,
                                         2.589151,
                                         1.156019,
                                         1.155781])
            self.user_config = {
                'Qweights_loadpath': '/local-scratch/xlv/SL_optCtrl/Qinit/dubinsCar/trained_model/Qf_weights.pkl'}
            with open(self.user_config['Qweights_loadpath'], 'rb') as wt_f:
                print("start re-initialize Q function from {}".format(self.user_config['Qweights_loadpath']))
                wt = pickle.load(wt_f)
                self.q[0].weight.data = torch.from_numpy(np.transpose(wt[0][0]))
                self.q[0].bias.data = torch.from_numpy(wt[0][1])

                self.q[2].weight.data = torch.from_numpy(np.transpose(wt[1][0]))
                self.q[2].bias.data = torch.from_numpy(wt[1][1])

                self.q[4].weight.data = torch.from_numpy(np.transpose(wt[2][0]))
                self.q[4].bias.data = torch.from_numpy(wt[2][1])
                print("weight re-initialize succeeds!")


    def forward(self, obs, act):
        if self.normmean is not None:
            # print("we are applying input normalization ...")
            tmp = (torch.cat([obs, act], dim=-1) - self.normmean) / self.normstd
            q = self.q(tmp)
        else:
            q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, initQ=False):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, initQ=initQ)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation, initQ=initQ)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

if __name__ == "__main__":
    # qfunc = MLPQFunction(obs_dim=11, act_dim=2, hidden_sizes=(256,256), activation=nn.ReLU)
    # print("Layer:", qfunc.q[0])
    # print("Layer shape:", np.shape(qfunc.q[0].weight))
    # print("Layer weight before:", qfunc.q[0].weight)

    ActorFunc = MLPActor(obs_dim=11, act_dim=2, hidden_sizes=(256,256), activation=nn.ReLU, act_limit=2)
    print(ActorFunc.pi[4].weight)
    print(ActorFunc.pi[4].bias)