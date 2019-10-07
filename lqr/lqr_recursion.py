#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
LQR Algorithm Recursion solver
"""
import pathlib
import sys

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
from chainer import functions as F
from util import bmv, bger, batch_lu_factor, batch_lu_solve
import copy
import chainer


class LqrRecursion:
    """
    LQR Recursion solver class
    State space : n dimensional
    State space : m dimensional

    """

    def __init__(self, x_init, C, c, large_f, f, T, n_state, n_ctrl, u_zero_Index=None):
        """ constructor

        :param x_init: initial state
        :param C: quadratic cost term (must be PSD)
        :param c: quadratic cost term
        :param large_f: affine cost term
        :param f: affine cost term
        :param T: horizon length
        :param n_state: dim of state
        :param n_ctrl: dim of control
        :param u_zero_Index: used only for backpropagation of mpc
        """
        self.x_init = x_init
        self.C = C
        self.c = c
        self.F = large_f
        self.f = f
        self.T = T
        self.n_batch = C.shape[1]
        self.n_state = n_state  # dim of state
        self.n_ctrl = n_ctrl
        self.n_sc = n_state + n_ctrl
        self.xp = chainer.backend.get_array_module(x_init)
        self.u_zero_Index = u_zero_Index
        assert list(self.x_init.shape) == [self.n_batch, self.n_state]
        assert list(self.C.shape) == [self.T, self.n_batch, self.n_sc, self.n_sc], "C dim mismatch"
        assert list(self.c.shape) == [self.T, self.n_batch, self.n_sc], str(self.c.shape) + \
                                                                        " c dim mismatch: expected " + str(
            [self.T, self.n_batch, self.n_sc])
        '''
        if list(self.F.shape)[0] == self.T:
            self.F = self.F[:self.T - 1]
        else:
            assert (self.F.shape[0]) == self.T - 1, "F dimension"
        assert list(self.F.shape) == [self.T - 1, self.n_batch, self.n_state, self.n_sc], str(
            self.F.shape) + " predicted:" + str(
            self.T - 1) + " " + str(self.n_batch) + " " + str(self.n_state) + " " + str(self.n_sc) + "F dim mismatch"
        '''
        if self.f is not None:
            assert list(self.f.shape) == [self.T - 1, self.n_batch, self.n_state], " f dim mismatch"
        self.u_zero_Index = u_zero_Index

    def backward(self):
        """ LQR backward recursion
        Note: Ks ks is reversed version fo original
        :return: Ks, ks gain
        """
        Ks = []
        ks = []
        Vt = None
        vt = None
        # self.T-1 to 0 loop
        for t in range(self.T - 1, -1, -1):
            # initial case
            if t == self.T - 1:
                Qt = self.C[t]
                qt = self.c[t]
            else:
                Ft = self.F[t]
                Ft_T = F.transpose(Ft, axes=(0, 2, 1))
                assert Ft.dtype.kind == 'f', "Ft dtype"
                assert Vt.dtype.kind == 'f', "Vt dtype"
                Qt = self.C[t] + F.matmul(F.matmul(Ft_T, Vt), Ft)
                if self.f is None:
                    # NOTE f.nelement() == 0 condition ?
                    qt = self.c[t] + bmv(Ft_T, vt)
                else:
                    # f is not none
                    ft = self.f[t]
                    qt = self.c[t] + bmv(F.matmul(Ft_T, Vt), ft) + bmv(Ft_T, vt)
            assert list(qt.shape) == [self.n_batch, self.n_sc], "qt dim mismatch"
            assert list(Qt.shape) == [self.n_batch, self.n_sc, self.n_sc], str(Qt.shape) + " Qt dim mismatch"
            Qt_xx = Qt[:, :self.n_state, :self.n_state]
            Qt_xu = Qt[:, :self.n_state, self.n_state:]
            Qt_ux = Qt[:, self.n_state:, :self.n_state]
            Qt_uu = Qt[:, self.n_state:, self.n_state:]
            qt_x = qt[:, :self.n_state]
            qt_u = qt[:, self.n_state:]
            assert list(Qt_uu.shape) == [self.n_batch, self.n_ctrl, self.n_ctrl], "Qt_uu dim mismatch"
            assert list(Qt_ux.shape) == [self.n_batch, self.n_ctrl, self.n_state], "Qt_ux dim mismatch"
            assert list(Qt_xu.shape) == [self.n_batch, self.n_state, self.n_ctrl], "Qt_xu dim mismatch"
            assert list(qt_x.shape) == [self.n_batch, self.n_state], "qt_x dim mismatch"
            assert list(qt_u.shape) == [self.n_batch, self.n_ctrl], "qt_u dim mismatch"
            # Next calculate Kt and kt
            # TODO LU decomposition
            if self.n_ctrl == 1 and self.u_zero_Index is None:
                # scalar
                Kt = - (1. / Qt_uu) * Qt_ux
                kt = - (1. / F.squeeze(Qt_uu, axis=2)) * qt_u
            elif self.u_zero_Index is None:
                # matrix
                Qt_uu_inv = F.batch_inv(Qt_uu)
                Kt = - F.matmul(Qt_uu_inv, Qt_ux)
                kt = - bmv(Qt_uu_inv, qt_u)
            else:
                # u_zero_index is not none
                index = self.u_zero_Index[t]
                qt_u_ = copy.deepcopy(qt_u)
                qt_u_ = F.where(index, self.xp.zeros_like(qt_u_.array), qt_u_)
                Qt_uu_ = copy.deepcopy(Qt_uu)
                notI = 1.0 - F.cast(index, qt_u_.dtype)
                Qt_uu_I = 1 - bger(notI, notI)
                Qt_uu_I = F.cast(Qt_uu_I, 'bool')
                Qt_uu_ = F.where(Qt_uu_I, self.xp.zeros_like(Qt_uu_.array), Qt_uu_)
                index_qt_uu = self.xp.array([self.xp.diagflat(index[i]) for i in range(index.shape[0])])
                Qt_uu_ = F.where(F.cast(index_qt_uu, 'bool'), Qt_uu + 1e-8, Qt_uu)
                Qt_ux_ = copy.deepcopy(Qt_ux)
                index_qt_ux = F.repeat(F.expand_dims(index, axis=2), Qt_ux.shape[2], axis=2)
                Qt_ux_ = F.where(index_qt_ux, self.xp.zeros_like(Qt_ux_.array), Qt_ux)
                #  print("qt_u_", qt_u_)
                #  print("Qt_uu_", Qt_uu_)
                #  print("Qt_ux_", Qt_ux_)
                if self.n_ctrl == 1:
                    Kt = - (1. / Qt_uu_) * Qt_ux_  # NOTE different from original
                    kt = - (1. / F.squeeze(Qt_uu_, axis=2)) * qt_u_
                else:
                    Qt_uu_LU_ = batch_lu_factor(Qt_uu_)
                    Kt = - batch_lu_solve(Qt_uu_LU_, Qt_ux_)
                    kt = - batch_lu_solve(Qt_uu_LU_, qt_u_)
            assert list(Kt.shape) == [self.n_batch, self.n_ctrl, self.n_state], "Kt dim mismatch"
            assert list(kt.shape) == [self.n_batch, self.n_ctrl], "kt dim mismatch"
            Kt_T = F.transpose(Kt, axes=(0, 2, 1))
            Ks.append(Kt)
            ks.append(kt)
            Vt = Qt_xx + F.matmul(Qt_xu, Kt) + F.matmul(Kt_T, Qt_ux) + F.matmul(F.matmul(Kt_T, Qt_uu), Kt)
            vt = qt_x + bmv(Qt_xu, kt) + bmv(Kt_T, qt_u) + bmv(F.matmul(Kt_T, Qt_uu), kt)

        assert len(Ks) == self.T, "Ks length error"

        Ks.reverse()
        ks.reverse()
        return Ks, ks

    def forward(self, Ks, ks):
        """ LQR forward recursion

        :param Ks: solved in backward recursion
        :param ks: solved in forward recursion
        :return: x, u
        """
        assert len(Ks) == self.T, "Ks length error"
        new_x = [self.x_init]
        new_u = []
        for t in range(self.T):
            Kt = Ks[t]
            kt = ks[t]
            xt = new_x[t]
            assert list(xt.shape) == [self.n_batch, self.n_state], str(xt.shape) + \
                                                                   " xt dim mismatch: expected" + str(
                [self.n_batch, self.n_state])
            new_ut = bmv(Kt, xt) + kt
            assert list(new_ut.shape) == [self.n_batch, self.n_ctrl], "actual:" + str(new_ut.shape)
            if self.u_zero_Index is not None:
                assert self.u_zero_Index[t].shape == new_ut.shape, str(self.u_zero_Index[t].shape) + " : " + str(
                    new_ut.shape)
                new_ut = F.where(self.u_zero_Index[t], self.xp.zeros_like(new_ut.array), new_ut)
                assert list(new_ut.shape) == [self.n_batch, self.n_ctrl], "actual:" + str(new_ut.shape)
            new_u.append(new_ut)
            if t < self.T - 1:
                assert list(xt.shape) == [self.n_batch, self.n_state], "actual:" + str(xt.shape)
                assert list(new_u[t].shape) == [self.n_batch, self.n_ctrl], "actual:" + str(new_u[t].shape)
                xu_t = F.concat((xt, new_u[t]), axis=1)
                x = bmv(self.F[t], xu_t)
                if self.f is not None:
                    x += self.f[t]
                assert list(x.shape) == [self.n_batch, self.n_state], str(x.shape) + \
                                                                      " x dim mismatch: expected" + str(
                    [self.n_batch, self.n_state])
                new_x.append(x)
        new_x = F.stack(new_x, axis=0)
        new_u = F.stack(new_u, axis=0)
        assert list(new_x.shape) == [self.T, self.n_batch, self.n_state], str(new_x.shape) + " new x dim mismatch"
        assert list(new_u.shape) == [self.T, self.n_batch, self.n_ctrl], "new u dim mismatch"
        return new_x, new_u

    def solve_recursion(self):
        """ backward and forward

        :return: optimal state and control
        """
        Ks, ks = self.backward()
        new_x, new_u = self.forward(Ks, ks)
        return new_x, new_u
