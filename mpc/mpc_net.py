#! usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
MPC network function
"""
import pathlib
import sys

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
import chainer
import numpy as np
from box_ddp import BoxDDP
from chainer import functions as F

from util import expand_time_batch, LinDx


class MpcNet_dx(chainer.Link):
    """
    MPC network dynamics is unknown
    """

    def __init__(self, T, u_lower, u_upper, n_batch, n_state, n_ctrl, seed, u_init, eps=1e-5, not_improved_lim=5,
                 line_search_decay=0.2, max_line_search_iter=10, best_cost_eps=1e-4, max_iter=10,
                 verbose=False, ilqr_verbose=False):
        """ LQR constructor

        :param T:
        :param n_batch:
        :param n_state:
        :param n_ctrl:
        :param n_sc:
        :param seed:
        """
        super().__init__()
        self.u_lower = u_lower
        self.u_upper = u_upper
        assert (u_lower.array <= u_upper.array).all(), " lower is larger than upper"
        self.T = T
        self.n_batch = n_batch
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_sc = self.n_ctrl + self.n_state
        self.u_init = u_init
        self.eps = eps
        self.not_improved_lim = not_improved_lim
        self.ls_decay = line_search_decay
        self.max_ls_iter = max_line_search_iter
        self.best_cost_eps = best_cost_eps
        self.max_iter = max_iter
        self.verbose = verbose
        self.ilqr_verbose = ilqr_verbose
        assert list(self.u_lower.shape) == [self.T, self.n_batch, self.n_ctrl], 'actual' + str(self.u_lower.shape)
        assert list(self.u_upper.shape) == [self.T, self.n_batch, self.n_ctrl], 'actual' + str(self.u_lower.shape)

        with self.init_scope():
            self.xp.random.seed(seed)
            alpha = 0.2
            A = self.xp.eye(n_state).astype('float') + alpha * self.xp.random.randn(n_state, n_state).astype('float')
            self.A = chainer.Parameter(A).reshape(n_state, n_state)
            B = self.xp.random.randn(n_state, n_ctrl).astype('float')
            self.B = chainer.Parameter(B).reshape(n_state, n_ctrl)
        self.mpc_layer = BoxDDP(T=self.T, u_lower=self.u_lower, u_upper=self.u_upper, n_batch=self.n_batch,
                                n_state=self.n_state, n_ctrl=self.n_ctrl, u_init=self.u_init, eps=self.eps,
                                not_improved_lim=self.not_improved_lim, line_search_decay=self.ls_decay,
                                max_line_search_iter=self.max_ls_iter,
                                best_cost_eps=self.best_cost_eps, max_iter=self.max_iter, verbose=self.verbose,
                                ilqr_verbose=self.ilqr_verbose)

    def forward(self, inputs):
        """ Forward propagation

        :param inputs: x_init, u_init, cost = inputs
        :return:
        """
        ab_cat = F.concat((self.A, self.B), axis=1)
        large_f_learner = expand_time_batch(ab_cat, self.T - 1, self.n_batch)
        assert list(large_f_learner.shape) == [self.T - 1, self.n_batch, self.n_state,
                                               self.n_sc], " Learner's F dimension mismatch"
        assert large_f_learner.dtype.kind == 'f', "dtype error"
        x_init, cost = inputs
        f = chainer.Variable(np.zeros((self.T - 1, self.n_batch, self.n_state), dtype=x_init.dtype))
        dynamics = LinDx(large_f_learner, f)
        res = self.mpc_layer((x_init, cost, dynamics))
        return res


class MpcNet_dx(chainer.Link):
    """
    MPC network dynamics is unknown
    """

    def __init__(self, T, u_lower, u_upper, n_batch, n_state, n_ctrl, seed, u_init, eps=1e-5, not_improved_lim=5,
                 line_search_decay=0.2, max_line_search_iter=10, best_cost_eps=1e-4, max_iter=10,
                 verbose=False, ilqr_verbose=False):
        """ LQR constructor

        :param T:
        :param n_batch:
        :param n_state:
        :param n_ctrl:
        :param n_sc:
        :param seed:
        """
        super().__init__()
        self.u_lower = u_lower
        self.u_upper = u_upper
        assert (u_lower.array <= u_upper.array).all(), " lower is larger than upper"
        self.T = T
        self.n_batch = n_batch
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_sc = self.n_ctrl + self.n_state
        self.u_init = u_init
        self.eps = eps
        self.not_improved_lim = not_improved_lim
        self.ls_decay = line_search_decay
        self.max_ls_iter = max_line_search_iter
        self.best_cost_eps = best_cost_eps
        self.max_iter = max_iter
        self.verbose = verbose
        self.ilqr_verbose = ilqr_verbose
        assert list(self.u_lower.shape) == [self.T, self.n_batch, self.n_ctrl], 'actual' + str(self.u_lower.shape)
        assert list(self.u_upper.shape) == [self.T, self.n_batch, self.n_ctrl], 'actual' + str(self.u_lower.shape)

        with self.init_scope():
            self.xp.random.seed(seed)
            alpha = 0.2
            A = self.xp.eye(n_state).astype('float') + alpha * self.xp.random.randn(n_state, n_state).astype('float')
            self.A = chainer.Parameter(A).reshape(n_state, n_state)
            B = self.xp.random.randn(n_state, n_ctrl).astype('float')
            self.B = chainer.Parameter(B).reshape(n_state, n_ctrl)
        self.mpc_layer = BoxDDP(T=self.T, u_lower=self.u_lower, u_upper=self.u_upper, n_batch=self.n_batch,
                                n_state=self.n_state, n_ctrl=self.n_ctrl, u_init=self.u_init, eps=self.eps,
                                not_improved_lim=self.not_improved_lim, line_search_decay=self.ls_decay,
                                max_line_search_iter=self.max_ls_iter,
                                best_cost_eps=self.best_cost_eps, max_iter=self.max_iter, verbose=self.verbose,
                                ilqr_verbose=self.ilqr_verbose)

    def forward(self, inputs):
        """ Forward propagation

        :param inputs: x_init, u_init, cost = inputs
        :return:
        """
        ab_cat = F.concat((self.A, self.B), axis=1)
        large_f_learner = expand_time_batch(ab_cat, self.T - 1, self.n_batch)
        assert list(large_f_learner.shape) == [self.T - 1, self.n_batch, self.n_state,
                                               self.n_sc], " Learner's F dimension mismatch"
        assert large_f_learner.dtype.kind == 'f', "dtype error"
        x_init, cost = inputs
        f = chainer.Variable(np.zeros((self.T - 1, self.n_batch, self.n_state), dtype=x_init.dtype))
        dynamics = LinDx(large_f_learner, f)
        res = self.mpc_layer((x_init, cost, dynamics))
        return res


class MpcNet_cost(chainer.Link):
    """
    MPC network dynamics is unknown
    """

    def __init__(self, T, u_lower, u_upper, n_batch, n_state, n_ctrl, seed, u_init, eps=1e-5, not_improved_lim=5,
                 line_search_decay=0.2, max_line_search_iter=10, best_cost_eps=1e-4, max_iter=10,
                 verbose=False, ilqr_verbose=False):
        """ LQR constructor

        :param T:
        :param n_batch:
        :param n_state:
        :param n_ctrl:
        :param n_sc:
        :param seed:
        """
        super().__init__()
        self.u_lower = u_lower
        self.u_upper = u_upper
        assert (u_lower.array <= u_upper.array).all(), " lower is larger than upper"
        self.T = T
        self.n_batch = n_batch
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_sc = self.n_ctrl + self.n_state
        self.u_init = u_init
        self.eps = eps
        self.not_improved_lim = not_improved_lim
        self.ls_decay = line_search_decay
        self.max_ls_iter = max_line_search_iter
        self.best_cost_eps = best_cost_eps
        self.max_iter = max_iter
        self.verbose = verbose
        self.ilqr_verbose = ilqr_verbose
        assert list(self.u_lower.shape) == [self.T, self.n_batch, self.n_ctrl], 'actual' + str(self.u_lower.shape)
        assert list(self.u_upper.shape) == [self.T, self.n_batch, self.n_ctrl], 'actual' + str(self.u_lower.shape)

        with self.init_scope():
            self.xp.random.seed(seed)
            alpha = 0.2
            A = self.xp.eye(n_state).astype('float') + alpha * self.xp.random.randn(n_state, n_state).astype('float')
            self.A = chainer.Parameter(A).reshape(n_state, n_state)
            B = self.xp.random.randn(n_state, n_ctrl).astype('float')
            self.B = chainer.Parameter(B).reshape(n_state, n_ctrl)
        self.mpc_layer = BoxDDP(T=self.T, u_lower=self.u_lower, u_upper=self.u_upper, n_batch=self.n_batch,
                                n_state=self.n_state, n_ctrl=self.n_ctrl, u_init=self.u_init, eps=self.eps,
                                not_improved_lim=self.not_improved_lim, line_search_decay=self.ls_decay,
                                max_line_search_iter=self.max_ls_iter,
                                best_cost_eps=self.best_cost_eps, max_iter=self.max_iter, verbose=self.verbose,
                                ilqr_verbose=self.ilqr_verbose)

    def forward(self, inputs):
        """ Forward propagation

        :param inputs: x_init, u_init, cost = inputs
        :return:
        """
        ab_cat = F.concat((self.A, self.B), axis=1)
        large_f_learner = expand_time_batch(ab_cat, self.T - 1, self.n_batch)
        assert list(large_f_learner.shape) == [self.T - 1, self.n_batch, self.n_state,
                                               self.n_sc], " Learner's F dimension mismatch"
        assert large_f_learner.dtype.kind == 'f', "dtype error"
        x_init, cost = inputs
        f = chainer.Variable(np.zeros((self.T - 1, self.n_batch, self.n_state), dtype=x_init.dtype))
        dynamics = LinDx(large_f_learner, f)
        res = self.mpc_layer((x_init, cost, dynamics))
        return res
