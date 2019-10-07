#! usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Differentiable LQR solver and LQRnet
"""
import pathlib
import sys

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
import chainer
import chainer.functions as F
import numpy as xp
from chainer import function_node
from chainer.utils import type_check
from lqr_recursion import LqrRecursion
from util import bger, bmv, expand_time_batch


class DiffLqr(function_node.FunctionNode):
    """
    Differentiable LQR as described in [1] module 1
    """

    def __init__(self, T, n_batch, n_state, n_ctrl):
        """ constructor

        :param T:
        :param n_batch:
        :param n_state:
        :param n_ctrl:
        """
        super().__init__()
        self.T = T
        self.n_batch = n_batch
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_sc = n_state + n_ctrl

    def check_type_forward(self, in_types):
        """ type check

        :param in_types:
        :return:
        """
        n_in = in_types.size()
        type_check.expect(n_in == 5)
        x_init, C, c, large_f, f = in_types
        '''
        print(x_init.dtype.kind)
        print(C.dtype.kind)
        print(c.dtype.kind)
        print(large_f.dtype.kind)
        '''
        type_check.expect(
            x_init.dtype.kind == 'f',
            C.dtype.kind == 'f',
            c.dtype.kind == 'f',
            large_f.dtype.kind == 'f',
        )
        if f is not None:
            type_check.expect(f.dtype.kind == 'f')

    def forward(self, inputs):
        """ forward pass

        :param inputs: x_init, params
        :return: solution of optimization problem
        """
        x_init, C, c, large_f, f = inputs
        self.retain_inputs((0, 1, 2, 3))
        lqr = LqrRecursion(x_init, C, c, large_f, f, self.T, self.n_state, self.n_ctrl)
        x, u = lqr.solve_recursion()
        self.retain_outputs((0, 1))
        return x.data, u.data

    def backward(self, target_input_indexes, grad_outputs):
        """ Backward Pass

        :param target_input_indexes:
        :param grad_outputs:
        :return:
        """
        # Forward 2 calculate dual variable with backward recursion, Equation (7) in [1]
        x_init, C, c, large_f = self.get_retained_inputs()
        C_Tx = C[self.T - 1, :, :self.n_state, :]
        c_Tx = c[self.T - 1, :, :self.n_state]
        assert list(C_Tx.shape) == [self.n_batch, self.n_state, self.n_sc]
        x, u = self.get_retained_outputs()
        taus = F.concat((x, u), axis=2).reshape(self.T, self.n_batch, self.n_sc)
        Lambda_T = bmv(C_Tx, taus[self.T - 1]) + c_Tx
        Lambdas = [Lambda_T]
        # backward recursion calculate dual variable
        for i in range(self.T - 2, -1, -1):
            Lambda_tp1 = Lambdas[self.T - 2 - i]
            tau_t = taus[i]
            F_t = large_f[i]
            F_tx_T = F.transpose(F_t[:, :self.n_state, :self.n_state], axes=(0, 2, 1))
            C_tx = C[i][:, :self.n_state, :]
            c_tx = c[i][:, :self.n_state]
            Lambda_t = bmv(F_tx_T, Lambda_tp1) + bmv(C_tx, tau_t) + c_tx
            Lambdas.append(Lambda_t)
        Lambdas.reverse()
        # Backward 1
        grad_x, grad_u = grad_outputs
        xp = chainer.backend.get_array_module(*grad_outputs)
        zero_init = chainer.Variable(xp.zeros_like(x_init.data))
        zero_f = chainer.Variable(xp.zeros((self.T - 1, self.n_batch, self.n_state)))
        drl = F.concat((grad_x, grad_u), axis=2).reshape(self.T, self.n_batch, self.n_sc)
        lqr = LqrRecursion(zero_init, C, drl, large_f, zero_f, self.T, self.n_state, self.n_ctrl)
        dx, du = lqr.solve_recursion()
        # Backward 2 calculate dual variable
        d_taus = F.concat((dx, du), axis=2).reshape(self.T, self.n_batch, self.n_sc)
        d_lambda_T = bmv(C_Tx, d_taus[self.T - 1]) + drl[self.T - 1][:, :self.n_state]
        d_lambdas = [d_lambda_T]
        for i in range(self.T - 2, -1, -1):
            d_lambda_tp1 = d_lambdas[self.T - 2 - i]
            d_tau_t = d_taus[i]
            F_t = large_f[i]
            F_tx_T = F.transpose(F_t[:, :self.n_state, :self.n_state], axes=(0, 2, 1))
            C_tx = C[i][:, :self.n_state, :]
            d_rl = drl[i][:, :self.n_state]
            d_lambda_t = bmv(F_tx_T, d_lambda_tp1) + bmv(C_tx, d_tau_t) + d_rl
            d_lambdas.append(d_lambda_t)
        d_lambdas.reverse()
        # Backward line 3 compute derivatives : Equation 8
        dC = F.stack([0.5 * bger(d_taus[t], taus[t]) + bger(taus[t], d_taus[t]) for t in range(self.T)], axis=0)
        dc = F.stack([d_taus[t] for t in range(self.T)], axis=0)
        dF = F.stack(
            [bger(d_lambdas[t + 1], taus[t]).data + bger(Lambdas[t + 1], d_taus[t]).data for t in range(self.T - 1)],
            axis=0)
        df = F.stack(d_lambdas[:self.T - 1], axis=0)
        d_x_init = d_lambdas[0]
        '''
        print(d_x_init.shape)
        print(dC.shape)
        print(dc.shape)
        print(dF.shape)
        print(df.shape)
        '''
        return d_x_init, dC, dc, dF, df


class LqrNet(chainer.Link):
    """LQR network chain
        Only dynamics is unknown
    """

    def __init__(self, T, n_batch, n_state, n_ctrl, seed):
        """ LQR constructor

        :param T:
        :param n_batch:
        :param n_state:
        :param n_ctrl:
        :param n_sc:
        :param seed:
        """
        super().__init__()
        self.T = T
        self.n_batch = n_batch
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_sc = self.n_ctrl + self.n_state
        with self.init_scope():
            xp.random.seed(seed)
            alpha = 0.2
            A = xp.eye(n_state).astype('f8') + alpha * xp.random.randn(n_state, n_state).astype('f8')
            self.A = chainer.Parameter(A).reshape(n_state, n_state)
            B = xp.random.randn(n_state, n_ctrl).astype('f8')
            self.B = chainer.Parameter(B).reshape(n_state, n_ctrl)
            self.lqr_layer = DiffLqr(self.T, self.n_batch, self.n_state, self.n_ctrl)

            '''
            alpha = 0.6
            self.A = chainer.Parameter(xp.array([[1.0, 0, 0], [1, 1.0, 0], [0, 1, 1]],
                                                dtype=xp.float64) + alpha * xp.random.randn(n_state, n_state).astype(
                'f8')).reshape(n_state, n_state)
            self.B = chainer.Parameter(xp.array([[1.0], [0], [0]],
                                                dtype=xp.float64).T.reshape(n_state, n_ctrl) + alpha * xp.random.randn(
                n_state, n_ctrl).astype('f8')).reshape(n_state, n_ctrl)
            self.lqr_layer = DiffLqr(self.T, self.n_batch, self.n_state, self.n_ctrl)
            '''

    def forward(self, inputs):
        """ Forward propagation

        :param inputs: x_init, params
        :return:
        """
        ab_cat = F.concat((self.A, self.B), axis=1)
        large_f_learner = expand_time_batch(ab_cat, self.T - 1, self.n_batch)
        assert list(large_f_learner.shape) == [self.T - 1, self.n_batch, self.n_state,
                                               self.n_sc], " Learner's F dimension mismatch"
        assert large_f_learner.dtype.kind == 'f', "dtype error"
        x_init, C, c, f = inputs
        return self.lqr_layer.apply((x_init, C, c, large_f_learner, f))


class LqrNet_cost_dx(chainer.Link):
    """LQR net cost and dynamics is unknown
    """

    def __init__(self, T, n_batch, n_state, n_ctrl, seed):
        """ LQR constructor

        :param T:
        :param n_batch:
        :param n_state:
        :param n_ctrl:
        :param n_sc:
        :param seed:
        """
        super().__init__()
        self.T = T
        self.n_batch = n_batch
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_sc = self.n_ctrl + self.n_state
        with self.init_scope():
            xp.random.seed(seed)
            alpha = 0.2
            A = xp.eye(n_state).astype('f8') + alpha * xp.random.randn(n_state, n_state).astype('f8')
            self.A = chainer.Parameter(A).reshape(n_state, n_state)
            B = xp.random.randn(n_state, n_ctrl).astype('f8')
            self.B = chainer.Parameter(B).reshape(n_state, n_ctrl)
            C = xp.eye(self.n_sc).astype('f8') + alpha * xp.random.randn(self.n_sc, self.n_sc).astype('f8')
            self.C = chainer.Parameter(C).reshape(self.n_sc, self.n_sc)
            c = xp.random.randn(self.n_sc).astype('f8')
            self.c = chainer.Parameter(c).reshape(self.n_sc)
            self.lqr_layer = DiffLqr(self.T, self.n_batch, self.n_state, self.n_ctrl)

    def forward(self, inputs):
        """ Forward propagation

        :param inputs: x_init, params
        :return:
        """
        ab_cat = F.concat((self.A, self.B), axis=1)
        large_f_learner = expand_time_batch(ab_cat, self.T - 1, self.n_batch)
        assert list(large_f_learner.shape) == [self.T - 1, self.n_batch, self.n_state,
                                               self.n_sc], " Learner's F dimension mismatch"
        assert large_f_learner.dtype.kind == 'f', "dtype error"
        x_init, f = inputs
        C = expand_time_batch(self.C, self.T, self.n_batch)
        c = expand_time_batch(self.c, self.T, self.n_batch)
        return self.lqr_layer.apply((x_init, C, c, large_f_learner, f))
