# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Imitation Learning 
#
# Learn {A, B} (dynamics) from imitation loss which is 

# \begin{equation*}
# Loss ={E}_{\text{x_init}} [\|\tau(x_\text{init}; \theta)- \tau(x_\text{init};\hat{\theta})|_2^2]
# \end{equation*}

# !pwd

train_seed = 1

# Cost はすでに求められていて部分的にBack Propagationをする
import chainer
import numpy as np
import sys
sys.path.append("/Users/i19_arahata/Program/chainer_mpc/mpc/")
from box_ddp import BoxDDP
from mpc_net import MpcNet_dx
from chainer import functions as F
from util import expand_time_batch, bmv, expand_batch, QuadCost, LinDx
import os
'''
import matplotlib.pyplot as plt
'''

T = 5
n_state = 3
n_ctrl = 3
n_sc = n_ctrl + n_state
n_batch = 128
dtype = np.float32
expert_seed = 42
np.random.seed(expert_seed)
alpha = 0.2

exp = dict(
    Q = chainer.Variable(np.eye(n_sc)), #Cost
    p = chainer.Variable(np.random.randn(n_sc)), # cost (little c)
    A = chainer.Variable(np.eye(n_state) + alpha * np.random.randn(n_state, n_state)), # F left side
    B = chainer.Variable(np.random.randn(n_state, n_ctrl))
)

exp_ab_cat = F.concat((exp['A'], exp['B']), axis=1)
exp_large_f = expand_time_batch(exp_ab_cat, T-1, n_batch)

exp['F'] = exp_large_f
exp['f'] = chainer.Variable(np.zeros((T - 1, n_batch, n_state), dtype=exp_large_f.dtype))
exp_C = expand_time_batch(exp['Q'], T, n_batch)
exp['C'] = exp_C
exp['c'] = expand_time_batch(exp['p'], T, n_batch)


u_lower = -10.0 * np.ones(n_ctrl)
u_upper = 10.0 *np.ones(n_ctrl)

u_lower = expand_time_batch(u_lower, T, n_batch)

u_upper = expand_time_batch(u_upper, T, n_batch)

true_cost = QuadCost(exp['C'], exp['c'])
true_dynamics = LinDx(exp['F'], exp['f'])


def get_loss(x_init):
    expert = BoxDDP(T, u_lower,u_upper, n_batch ,n_state,n_ctrl,None)
    with chainer.no_backprop_mode():
        x_true, u_true,_ = expert.forward((x_init.array, true_cost, true_dynamics))
    x_true = x_true.array
    u_true = u_true.array
    x_pred, u_pred,_ = net((x_init, true_cost))
    trajectory_loss = F.mean((u_true - u_pred)**2)
    trajectory_loss += F.mean((x_true - x_pred)**2)
    return trajectory_loss


opt = chainer.optimizers.RMSprop(lr=1e-2,alpha=0.5)
net = MpcNet_dx(T, u_lower, u_upper, n_batch, n_state, n_ctrl, train_seed, u_init = None, max_iter =10, verbose = True)
opt.setup(net)
fname =str(train_seed)+'_new_losses.csv'
loss_f = open(fname, 'w')
loss_f.write('im_loss,mse\n')
loss_f.flush()

print(F.mean(exp['A']).array ,F.mean(exp['B']).array)

# %%time
for i in range(10):
    net.cleargrads()
    x_init = chainer.Variable(np.random.randn(n_batch, n_state))
    loss = get_loss(x_init)
    loss.backward()
    opt.update()
    model_loss = F.mean((net.A - exp['A'])**2) + F.mean((net.B - exp['B'])**2)
    loss_f.write('{},{}\n'.format(loss.data, model_loss.data))
    loss_f.flush()
    print("iteration", i,  "{0:04f}".format(loss.data), "dyanmics loss ", "{0:04f}".format(model_loss.data))

# !pwd


