# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
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

# +
# Cost はすでに求められていて部分的にBack Propagationをする
import chainer
import numpy as np
import sys
sys.path.append("../lqr")
from lqr_recursion import LqrRecursion
from differentiable_lqr import  DiffLqr, LqrNet
from chainer import functions as F
import chainer.computational_graph as c

'''
import matplotlib.pyplot as plt
'''
# -

T = 51
f = None
n_state = 3
n_ctrl = 1
n_sc = n_ctrl + n_state
n_batch = 128
dtype = np.float64
# expert parameters
exp = dict(
    A = chainer.Variable(np.array([[np.array([[
        1.0, 0, 0],
        [1, 1.0, 0],
        [0, 1, 1]]) for i in range(n_batch)]
        for j in range(T)],dtype = dtype)).reshape(T, n_batch, n_state, n_state),
    B = chainer.Variable(np.array([[np.array([[
         1.0],
        [ 0],
        [ 0]]) for i in range(n_batch)]
        for j in range(T)],dtype = dtype)).reshape(T, n_batch, n_state, n_ctrl),
    F=chainer.Variable(np.array([[np.array([[
        1.0, 0, 0, 1],
        [1, 1.0, 0, 0],
        [0, 1, 1, 0]]) for i in range(n_batch)]
        for j in range(T)],dtype = dtype)).reshape(T, n_batch, n_state, n_sc),
    c=chainer.Variable(np.array([[np.array([0, 0, 0.0, 0]).T for i in range(n_batch)]for j in range(T)],dtype=dtype)).reshape(T, n_batch, n_sc),
    f=None,
    C=chainer.Variable(np.array([[np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1]]) for i in range(n_batch)]for j in range(T)],dtype=dtype)).reshape(T, n_batch, n_sc, n_sc)
)

print(exp['C'].dtype.kind)
print(exp['c'].dtype.kind)
print(exp['F'].dtype.kind)


def get_loss(x_init):
    expert = LqrRecursion(x_init, exp['C'], exp['c'], exp['F'], exp['f'], T, n_state, n_ctrl)
    x_true, u_true = expert.solve_recursion()    
    x_pred, u_pred = net((x_init, exp['C'], exp['c'], exp['f']))
    # print(u_pred.dtype)
    g = c.build_computational_graph((x_pred), remove_split=True)
    """
    print(g.nodes)
    for i in range(len(g.nodes)):
        print(type(g.nodes[i]))
        if 'name' in dir(g.nodes[i]):
            print(g.nodes[i].name)
    """
    """
    with open('grapht.dot', 'w') as o:
        tmp = g.dump()
        o.write(tmp)
    """
    trajectory_loss = F.mean((u_true - u_pred)**2) + F.mean((x_true - x_pred)**2)
    
    return trajectory_loss

opt = chainer.optimizers.RMSprop(lr=1e-2)
net = LqrNet(T, n_batch, n_state, n_ctrl, 0)
opt.setup(net)

for i in range(5000):
    net.cleargrads()
    x_init = chainer.Variable(np.random.randn(n_batch, n_state))
    loss = get_loss(x_init)
    loss.backward()
    opt.update()
    model_loss = F.mean((net.A - exp['A'])**2) + F.mean((net.B - exp['B'])**2)
    plot_interval = 100
    if i % plot_interval == 0:
        print("train loss ", "{0:04f}".format(loss.data), "dyanmics loss ", "{0:04f}".format(model_loss.data))




