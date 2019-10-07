#! usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
MPC network function
"""
import chainer
import numpy as np
from chainer import functions as F


class Pendulum_Net_cost_logit(chainer.Link):
    """
    Pendulum network
    parameter of cost is logit
    """

    def __init__(self, n_sc):
        super().__init__()
        self.n_sc = n_sc
        with self.init_scope():
            learn_q_logit = self.xp.zeros(self.n_sc)
            learn_p = self.xp.zeros(self.n_sc)
            self.learn_q_logit = chainer.Parameter(learn_q_logit)
            self.learn_p = chainer.Parameter(learn_p)

    def forward(self, xinit, env, train_warm_start_idxs):
        """ forward function
        :param xinit:
        :param env:Pendulum_DX
        :param train_warm_start_idxs:
        :return:
        """
        q = F.sigmoid(self.learn_q_logit)
        p = F.sqrt(q) * self.learn_p
        # p = self.learn_p
        x_mpc, u_mpc = env.mpc(env.true_dx, xinit, q, p,
                               u_init=self.xp.transpose(train_warm_start_idxs, axes=(1, 0, 2)))
        return x_mpc, u_mpc

class Pendulum_Net_cost_lower_triangle(chainer.Link):
    """
    Pendulum network
    parameter of cost is LL^T
    """

    def __init__(self, n_sc, isrand=False):
        super().__init__()
        self.n_sc = n_sc
        with self.init_scope():
            if isrand:
                learn_q_logit = self.xp.random.rand(self.n_sc)
                learn_p = self.xp.random.rand(self.n_sc)
            else:
                learn_q_logit = self.xp.zeros(self.n_sc)
                learn_p = self.xp.zeros(self.n_sc)
            self.learn_q_logit = chainer.Parameter(learn_q_logit)
            self.learn_p = chainer.Parameter(learn_p)
            num_rand = int(self.n_sc * (self.n_sc-1)/2)
            self.lower_without_diag = chainer.Parameter(self.xp.zeros(num_rand))
            # self.n_s = self.n_sc - 1
            # num_rand = int(self.n_s * (self.n_s-1)/2)
            # self.lower_without_diag = chainer.Parameter(self.xp.zeros(num_rand))

    def forward(self, xinit, env, train_warm_start_idxs):
        """ forward function
        :param xinit:
        :param env:Pendulum_DX
        :param train_warm_start_idxs:
        :return:
        """
        Q = self.xp.zeros((self.n_sc, self.n_sc))
        index_diag = self.xp.zeros((self.n_sc, self.n_sc), dtype=bool)
        self.xp.fill_diagonal(index_diag, True)
        index_not_diag = self.xp.zeros((self.n_sc, self.n_sc), dtype=bool)
        index_not_diag[self.xp.tril_indices(self.n_sc, -1)] = True
        Q = F.scatter_add(Q, self.xp.tril_indices(self.n_sc, -1), self.lower_without_diag)
        diag_q = F.sigmoid(self.learn_q_logit)
        Q = F.where(index_diag, diag_q, Q)
        Q = Q @ Q.T
        p = self.learn_p
        # print("p.shape", p.shape)
        # print(self.n_sc)
        # p = F.concat((p, self.xp.array(0.0).reshape(1,)), axis=0)
        # print("Q ", Q)
        # print("p", p)
        x_mpc, u_mpc = env.mpc_Q(env.true_dx, xinit, Q, p,
                               u_init=self.xp.transpose(train_warm_start_idxs, axes=(1, 0, 2)))
        return x_mpc, u_mpc


OBSERVATION_MATRIX = np.array([[0., 4., 1., 0.], [1., 0., 4., 0.], [0., 4., 0., 0.], [0., 0., 0., 1.]])
"""
np.linalg.inv(transmat.T) @ np.diagflat(q)@ np.linalg.inv(transmat)
array([[ 1.61000e+01, -4.00000e+00, -1.61000e+01,  0.00000e+00],
       [-4.00000e+00,  1.00000e+00,  4.00000e+00,  0.00000e+00],
       [-1.61000e+01,  4.00000e+00,  1.61625e+01,  0.00000e+00],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e-03]])
       
"""

class Pendulum_Net_cost_logit_strange_obervation(chainer.Link):
    """
    Pendulum network
    parameter of cost is logit
    """

    def __init__(self, n_sc, isrand=False):
        super().__init__()
        self.n_sc = n_sc
        with self.init_scope():
            if isrand:
                self.xp.random.seed(0)
                learn_q_logit = self.xp.random.rand(self.n_sc)
                learn_p = self.xp.random.rand(self.n_sc)
            else:
                learn_q_logit = self.xp.zeros(self.n_sc)
                learn_p = self.xp.zeros(self.n_sc)
            self.learn_q_logit = chainer.Parameter(learn_q_logit)
            self.learn_p = chainer.Parameter(learn_p)

    def forward(self, xinit, env, train_warm_start_idxs):
        """ forward function
        :param xinit:
        :param env:Pendulum_DX
        :param train_warm_start_idxs:
        :return:
        """
        q = F.sigmoid(self.learn_q_logit)
        p = F.sqrt(q) * self.learn_p
        # p = self.learn_p
        Q = self.xp.zeros((self.n_sc, self.n_sc))
        index_diag = self.xp.zeros((self.n_sc, self.n_sc), dtype=bool)
        self.xp.fill_diagonal(index_diag, True)
        Q = F.where(index_diag, q, Q)
        Q = OBSERVATION_MATRIX.T @ Q @ OBSERVATION_MATRIX
        p = p @ OBSERVATION_MATRIX
        x_mpc, u_mpc = env.mpc_Q(env.true_dx, xinit, Q, p,
                               u_init=self.xp.transpose(train_warm_start_idxs, axes=(1, 0, 2)))
        return x_mpc, u_mpc


class Pendulum_Net_cost_lower_triangle_strange_obervation(chainer.Link):
    """
    Pendulum network
    parameter of cost is LL^T
    """

    def __init__(self, n_sc, isrand=False):
        super().__init__()
        self.n_sc = n_sc
        with self.init_scope():
            if isrand:
                self.xp.random.seed(0)
                learn_q_logit = self.xp.random.rand(self.n_sc)
                learn_p = self.xp.random.rand(self.n_sc)
            else:
                learn_q_logit = self.xp.zeros(self.n_sc)
                learn_p = self.xp.zeros(self.n_sc)
            self.learn_q_logit = chainer.Parameter(learn_q_logit)
            self.learn_p = chainer.Parameter(learn_p)
            num_rand = int(self.n_sc * (self.n_sc-1)/2)
            self.lower_without_diag = chainer.Parameter(self.xp.zeros(num_rand))

    def forward(self, xinit, env, train_warm_start_idxs):
        """ forward function
        :param xinit:
        :param env:Pendulum_DX
        :param train_warm_start_idxs:
        :return:
        """
        Q = self.xp.zeros((self.n_sc, self.n_sc))
        index_diag = self.xp.zeros((self.n_sc, self.n_sc), dtype=bool)
        self.xp.fill_diagonal(index_diag, True)
        index_not_diag = self.xp.zeros((self.n_sc, self.n_sc), dtype=bool)
        index_not_diag[self.xp.tril_indices(self.n_sc, -1)] = True
        Q = F.scatter_add(Q, self.xp.tril_indices(self.n_sc, -1), self.lower_without_diag)
        diag_q = F.sigmoid(self.learn_q_logit)
        Q = F.where(index_diag, diag_q, Q)
        Q = Q @ Q.T
        p = self.learn_p
        Q = OBSERVATION_MATRIX.T @ Q @ OBSERVATION_MATRIX
        p = p @ OBSERVATION_MATRIX
        x_mpc, u_mpc = env.mpc_Q(env.true_dx, xinit, Q, p,
                               u_init=self.xp.transpose(train_warm_start_idxs, axes=(1, 0, 2)))
        return x_mpc, u_mpc