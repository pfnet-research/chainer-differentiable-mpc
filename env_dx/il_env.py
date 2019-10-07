#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Imitation learning environment
"""

import pathlib
# import cupy as xp
import sys

import numpy as xp

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../mpc')
sys.path.append(str(current_dir) + '/../')
from box_ddp import BoxDDP
from pendulum import PendulumDx
from chainer import functions as F
from util import QuadCost, chainer_diag


class IL_Env:
    """
    Imitation learning Environmn class
    """

    def __init__(self, env, lqr_iter=500, mpc_T=20):
        """

        :param env:
        :param lqr_iter:
        :param mpc_T:
        """
        self.env = env
        if self.env == 'pendulum':
            self.true_dx = PendulumDx()
        else:
            assert False

        self.lqr_iter = lqr_iter
        self.mpc_T = mpc_T
        self.train_data = None
        self.val_data = None
        self.test_data = None

    @staticmethod
    def sample_xinit(n_batch=1):
        """ random sampling x_init

        :param n_batch:
        :return:
        """

        def uniform(shape, low, high):
            """

            :param shape:
            :param low:
            :param high:
            :return:
            """
            r = high - low
            return xp.random.rand(shape) * r + low

        th = uniform(n_batch, -(1 / 2) * xp.pi, (1 / 2) * xp.pi)
        # th = uniform(n_batch, -xp.pi, xp.pi)
        thdot = uniform(n_batch, -1., 1.)
        xinit = xp.stack((xp.cos(th), xp.sin(th), thdot), axis=1)
        return xinit

    def populate_data(self, n_train, n_val, n_test, seed=0):
        """

        :param n_train:
        :param n_val:
        :param n_test:
        :param seed:
        :return:
        """
        xp.random.seed(seed)
        n_data = n_train + n_val + n_test
        xinit = self.sample_xinit(n_batch=n_data)
        print(xinit.shape)
        # for (1,0,0) into the dataset
        '''
        n_init_zero = int(n_train/4)
        xinit[n_init_zero][0] = 1.0
        xinit[n_init_zero][1] = 0.0
        xinit[n_init_zero][2] = 0.0
        '''
        true_q, true_p = self.true_dx.get_true_obj()
        # self.mpc defined later
        true_x_mpc, true_u_mpc = self.mpc(self.true_dx, xinit, true_q, true_p, update_dynamics=True)
        true_x_mpc = true_x_mpc.array
        true_u_mpc = true_u_mpc.array
        tau = xp.concatenate((true_x_mpc, true_u_mpc), axis=2)
        tau = xp.transpose(tau, (1, 0, 2))

        self.train_data = tau[:n_train]
        self.val_data = tau[n_train:n_train + n_val]
        self.test_data = tau[-n_test:]

    def mpc(self, dx, xinit, q, p, u_init=None, eps_override=None,
            lqr_iter_override=None, update_dynamics=False):
        """

        :param dx:
        :param xinit:
        :param q:
        :param p:
        :param u_init:
        :param eps_override:
        :param lqr_iter_override:
        :return:
        """
        n_batch = xinit.shape[0]
        n_sc = self.true_dx.n_state + self.true_dx.n_ctrl

        Q = chainer_diag(q)
        Q = F.expand_dims(Q, axis=0)
        Q = F.expand_dims(Q, axis=0)
        Q = F.repeat(Q, self.mpc_T, axis=0)
        Q = F.repeat(Q, n_batch, axis=1)

        p = F.expand_dims(p, axis=0)
        p = F.expand_dims(p, axis=0)
        p = F.repeat(p, self.mpc_T, axis=0)
        p = F.repeat(p, n_batch, axis=1)
        if eps_override:
            eps = eps_override
        else:
            eps = self.true_dx.mpc_eps

        if lqr_iter_override:
            lqr_iter = lqr_iter_override
        else:
            lqr_iter = self.lqr_iter
        assert len(Q.shape) == 4
        assert len(p.shape) == 3
        solver = BoxDDP(
            T=self.mpc_T, u_lower=self.true_dx.lower, u_upper=self.true_dx.upper,
            n_batch=n_batch, n_state=self.true_dx.n_state, n_ctrl=self.true_dx.n_ctrl,
            u_init=u_init, eps=eps, max_iter=lqr_iter, verbose=False,
            exit_unconverged=False, detach_unconverged=True,
            line_search_decay=self.true_dx.linesearch_decay,
            max_line_search_iter=self.true_dx.max_linesearch_iter,
            update_dynamics=update_dynamics
        )

        x_mpc, u_mpc, objs_mpc = solver((xinit, QuadCost(Q, p), dx))
        '''
        g = c.build_computational_graph(u_mpc)
        with open('graph.dot', 'w') as o:
            o.write(g.dump())
        assert False
        '''
        return x_mpc, u_mpc

    def mpc_Q(self, dx, xinit, Q, p, u_init=None, eps_override=None,
              lqr_iter_override=None, update_dynamics=False):
        """

        :param dx:
        :param xinit:
        :param q:
        :param p:
        :param u_init:
        :param eps_override:
        :param lqr_iter_override:
        :return:
        """
        n_batch = xinit.shape[0]
        n_sc = self.true_dx.n_state + self.true_dx.n_ctrl

        Q = F.expand_dims(Q, axis=0)
        Q = F.expand_dims(Q, axis=0)
        Q = F.repeat(Q, self.mpc_T, axis=0)
        Q = F.repeat(Q, n_batch, axis=1)

        p = F.expand_dims(p, axis=0)
        p = F.expand_dims(p, axis=0)
        p = F.repeat(p, self.mpc_T, axis=0)
        p = F.repeat(p, n_batch, axis=1)
        if eps_override:
            eps = eps_override
        else:
            eps = self.true_dx.mpc_eps

        if lqr_iter_override:
            lqr_iter = lqr_iter_override
        else:
            lqr_iter = self.lqr_iter
        assert len(Q.shape) == 4
        assert len(p.shape) == 3
        solver = BoxDDP(
            T=self.mpc_T, u_lower=self.true_dx.lower, u_upper=self.true_dx.upper,
            n_batch=n_batch, n_state=self.true_dx.n_state, n_ctrl=self.true_dx.n_ctrl,
            u_init=u_init, eps=eps, max_iter=lqr_iter, verbose=False,
            exit_unconverged=False, detach_unconverged=True,
            line_search_decay=self.true_dx.linesearch_decay,
            max_line_search_iter=self.true_dx.max_linesearch_iter,
            update_dynamics=update_dynamics
        )

        x_mpc, u_mpc, objs_mpc = solver((xinit, QuadCost(Q, p), dx))
        '''
        g = c.build_computational_graph(u_mpc)
        with open('graph.dot', 'w') as o:
            o.write(g.dump())
        assert False
        '''
        return x_mpc, u_mpc
