#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
MPC_step function described in [1] Algorithm2
n_ctrl>1 and true cost and dynamics is QuadCost and LinDx test is ended
"""
import pathlib
import sys

from pnqp import PNQP

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
sys.path.append(str(current_dir) + '/../lqr')
import chainer
from util import QuadCost, LinDx
from util import to_xp, xpbger, xpbmv, xpbatch_lu_solve, xpget_cost, xpclamp, xpbquad, xpbdot
import copy
from collections import namedtuple
from chainer import function_node
from active_constrained_lqr import LQR_active

# back ward recursion output
LqrBackOut = namedtuple('lqrBackOut', 'n_total_qp_iter')
# forward recursion output
LqrForOut = namedtuple(
    'lqrForOut',
    'objs full_du_norm alpha_du_norm mean_alphas costs'
)


class MPCstep(function_node.FunctionNode):
    """ MPC forward backward calculation"""

    def __init__(self, controls, T, u_upper, u_lower, n_batch, n_state, n_ctrl, current_states,
                 true_cost, true_dynamics, ls_decay, max_ls_iter, verbose=False, need_expand=False,
                 no_op_forward=False):
        """" constructor
        :param controls: control
        :param u_upper: upper limit of control
        :param u_lower: lower limit of control
        :param n_batch: number of batch
        :param n_state:  dim of state
        :param n_ctrl: dim of control
        :param verbose:
        :return:
        """
        super().__init__()
        self.controls = to_xp(controls)
        self.u_upper = to_xp(u_upper)
        self.u_lower = to_xp(u_lower)
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_sc = self.n_state + self.n_ctrl
        self.n_batch = n_batch
        self.T = T
        self.verbose = verbose
        self.xp = chainer.backend.get_array_module(controls)
        self.back_out = None
        self.for_out = None
        self.current_states = to_xp(current_states)
        self.true_cost = true_cost  # not taylor expanded cost
        self.true_dynamics = true_dynamics
        self.need_expand = need_expand
        self.ls_decay = ls_decay
        self.max_ls_iter = max_ls_iter
        self.no_op_forward = no_op_forward

    def backward_rec(self, C_hat, c_hat, F_hat, f_hat):
        """ Back ward recursion over the linearized trajectory

        :param C_hat: approximated C
        :param c_hat: approximated c
        :param F_hat: approximated F
        :param f_hat: approximated f
        :return:
        """
        assert list(C_hat.shape) == [self.T, self.n_batch, self.n_sc, self.n_sc], \
            "C hat dim mismatch"
        assert list(c_hat.shape) == [self.T, self.n_batch, self.n_sc], \
            str(c_hat.shape) + " c hat dim mismatch: expected " + str([self.T, self.n_batch, self.n_sc])
        if list(F_hat.shape)[0] == self.T:
            F_hat = F_hat[:self.T - 1]
        else:
            assert (F_hat.shape[0]) == self.T - 1, "F_hat dimension"
        assert list(F_hat.shape) == [self.T - 1, self.n_batch, self.n_state, self.n_sc], \
            str(F_hat.shape) + " predicted:" + str(self.T - 1) + " " + \
            str(self.n_batch) + " " + str(self.n_state) + " " + str(self.n_sc) + "F_hat dim mismatch"
        if f_hat is not None:
            assert list(f_hat.shape) == [self.T - 1, self.n_batch, self.n_state] or \
                   list(f_hat.shape) == [self.T, self.n_batch, self.n_state], " f_hat dim mismatch"

        # Ks = []
        # ks = []
        Ks = self.xp.zeros((self.T, self.n_batch, self.n_ctrl, self.n_state))
        ks = self.xp.zeros((self.T, self.n_batch, self.n_ctrl))
        Vt = None
        vt = None
        prev_kt = None  # used for warm start up in Projected newton quadratic programmig
        n_total_qp_iter = 0
        # self.T-1 to 0 loop
        for t in range(self.T - 1, -1, -1):
            if t == self.T - 1:
                Qt = C_hat[t]
                qt = c_hat[t]
            else:
                Ft_hat = F_hat[t]
                Ft_hat_T = self.xp.transpose(Ft_hat, axes=(0, 2, 1))
                Qt = C_hat[t] + Ft_hat_T @ Vt @ Ft_hat
                if f_hat is None:
                    qt = c_hat[t] + xpbmv(Ft_hat_T, vt)
                else:
                    # f is not none
                    ft = f_hat[t]
                    qt = c_hat[t] + xpbmv(Ft_hat_T @ Vt, ft) + xpbmv(Ft_hat_T, vt)
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
            # calculate K and k
            # different from LQR case starts from here
            # lower_bound of control - current control
            assert not self.xp.isnan(self.controls[t]).any(), str(self.controls[t])
            assert not self.xp.isnan(self.u_lower[t]).any()
            assert not self.xp.isnan(self.u_upper[t]).any()
            lower_bound = self.u_lower[t] - self.controls[t]
            # upper_bound of control - current control
            upper_bound = self.u_upper[t] - self.controls[t]
            assert (lower_bound <= upper_bound).all(), " lower is larger than upper" \
                                                       + " lower: " + str(lower_bound) + "upper: " + str(upper_bound)
            kt, Qt_uu_free_LU, Index_free, n_qp_iter = PNQP(Qt_uu, qt_u, lower_bound, upper_bound,
                                                            x_init=prev_kt, n_iter=20)
            if self.verbose is True:
                print('  + n_qp_iter in mpc step: ', n_qp_iter + 1)
            n_total_qp_iter += 1 + n_qp_iter
            prev_kt = kt
            Qt_ux_copy = copy.deepcopy(Qt_ux)
            Index_Qt_ux_free = self.xp.repeat(self.xp.expand_dims((1.0 - Index_free), axis=2), self.n_state, axis=2)
            Index_Qt_ux_free = Index_Qt_ux_free.astype('bool')
            Qt_ux_copy[Index_Qt_ux_free] = 0.0
            # Qt_ux_copy = F.where(Index_Qt_ux_free, self.xp.zeros_like(Qt_ux.data), Qt_ux_copy)
            if self.n_ctrl == 1:
                # Bad naming, Qt_uu_free_LU is scalar
                Kt = -((1. / Qt_uu_free_LU) * Qt_ux_copy)
            else:
                # Qt_uu K_{t,f} = - Qt_ux
                Kt = - xpbatch_lu_solve(Qt_uu_free_LU, Qt_ux_copy)
            assert list(Kt.shape) == [self.n_batch, self.n_ctrl, self.n_state], "Kt dim mismatch"
            assert list(kt.shape) == [self.n_batch, self.n_ctrl], "kt dim mismatch"
            Kt_T = self.xp.transpose(Kt, axes=(0, 2, 1))
            assert not self.xp.isnan(kt).any()
            assert not self.xp.isnan(Kt).any()
            Ks[t] = Kt
            ks[t] = kt
            Vt = Qt_xx + Qt_xu @ Kt + Kt_T @ Qt_ux + Kt_T @ Qt_uu @ Kt
            vt = qt_x + xpbmv(Qt_xu, kt) + xpbmv(Kt_T, qt_u) + xpbmv((Kt_T @ Qt_uu), kt)

        assert len(Ks) == self.T, "Ks length error"
        '''
        Ks.reverse()
        ks.reverse()
        '''
        return Ks, ks, LqrBackOut(n_total_qp_iter=n_total_qp_iter)

    def forward_rec(self, Ks, ks, true_cost, true_dynamics, ls_decay, max_ls_iter):
        """ Forward recursion and line search


        :param Ks:
        :param ks:
        :param true_cost: true Cost function
        :param true_dynamics: true dynamics function
        :param states: states_{1:T} current state iterate
        :param ls_decay: line search decay ratio
        :param max_ls_iter: max line search iteration
        :return:
        """
        assert len(Ks) == self.T, "Ks length error"
        states = self.current_states
        alphas = self.xp.ones(self.n_batch, dtype=self.controls.dtype)
        OLD_COST = xpget_cost(self.T, self.controls, true_cost, true_dynamics, x=states)
        current_cost = None
        n_iter = 0  # number of line search iterations
        full_du_norm = None  # initial change of u
        # line search terminate condition, alpha for all batch is decreased until all batch meets terminal condition.
        while (n_iter < max_ls_iter and current_cost is None or (current_cost > OLD_COST).any()):
            assert type(alphas) == self.xp.ndarray, "alphas dtype error"
            new_x = [states[0]]
            new_u = []
            dx = [self.xp.zeros_like(states[0])]
            objs = []  # cost
            for t in range(self.T):
                Kt = Ks[t]
                kt = ks[t]
                new_xt = new_x[t]
                xt = states[t]
                ut = self.controls[t]
                dxt = dx[t]
                new_ut = xpbmv(Kt, dxt) + ut

                assert not self.xp.isnan(new_ut).any()
                assert not self.xp.isinf(new_ut).any()
                xp_alpha = self.xp.diagflat(alphas).astype(dtype=kt.dtype)
                assert not self.xp.isnan(xp_alpha).any()
                assert not self.xp.isinf(xp_alpha).any()
                assert not self.xp.isnan(kt).any()
                add_new_ut = xp_alpha @ kt
                assert not self.xp.isnan(add_new_ut).any(), str(alphas) + " @" + str(kt) + "=" + str(add_new_ut)
                new_ut += add_new_ut
                assert not self.xp.isnan(new_ut).any(), str(xp_alpha) + str(kt)
                new_ut = xpclamp(new_ut, self.u_lower[t], self.u_upper[t])
                # delta_u is None
                assert not self.xp.isnan(new_ut).any()
                new_u.append(new_ut)
                new_xut = self.xp.concatenate((new_xt, new_ut), axis=1)
                if t < self.T - 1:
                    # Calculate next x_{t+1}
                    # Dynamics is linear
                    if isinstance(true_dynamics, LinDx):
                        large_f, f = true_dynamics.F, true_dynamics.f
                        large_f = to_xp(large_f)
                        f = to_xp(f)
                        # new_x_{t+1}
                        new_xtp1 = xpbmv(large_f[t], new_xut)
                        if f is not None:
                            new_xtp1 += f[t]
                    else:
                        # Dynamics is non linear
                        new_xtp1 = true_dynamics(new_xt, new_ut)
                        new_xtp1 = to_xp(new_xtp1)
                    assert not self.xp.isnan(new_xtp1).any()
                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - states[t + 1])
                # Calculate cost
                # If cost is quadratic
                if isinstance(true_cost, QuadCost):
                    C = true_cost.C
                    c = true_cost.c
                    C = to_xp(C)
                    c = to_xp(c)
                    obj = 0.5 * xpbquad(new_xut, C[t]) + xpbdot(new_xut, c[t])
                else:
                    obj = true_cost(new_xut)
                objs.append(obj)
            objs = self.xp.stack(objs, axis=0)
            current_cost = self.xp.sum(objs, axis=0)
            new_x = self.xp.stack(new_x, axis=0)
            new_u = self.xp.stack(new_u, axis=0)
            # only update once
            if full_du_norm is None:
                du = self.controls - new_u
                du = self.xp.transpose(du, axes=(0, 2, 1)).reshape(self.n_batch, self.T * self.n_ctrl)
                full_du_norm = self.xp.sqrt(self.xp.sum(du ** 2, axis=1))
            assert list(new_x.shape) == [self.T, self.n_batch, self.n_state], str(new_x.shape) + " new x dim mismatch"
            assert list(new_u.shape) == [self.T, self.n_batch, self.n_ctrl], "new u dim mismatch"
            index_decay = current_cost > OLD_COST
            assert not self.xp.isinf(alphas).any()
            alphas[index_decay] *= ls_decay
            assert not self.xp.isinf(alphas).any(), str(ls_decay)
            n_iter += 1
        # TODO Check this decay
        # If the iteration limit is hit, some alphas
        # are one step too small.
        alphas[current_cost > OLD_COST] /= ls_decay
        du = self.controls - new_u
        du = self.xp.transpose(du, axes=(0, 2, 1)).reshape(self.n_batch, self.T * self.n_ctrl)
        alpha_du_norm = self.xp.sqrt(self.xp.sum(du ** 2, axis=1))
        res = LqrForOut(
            objs, full_du_norm,
            alpha_du_norm,
            self.xp.mean(alphas),
            current_cost
        )
        assert not self.xp.isnan(new_x).any()
        assert not self.xp.isnan(new_u).any()
        return new_x, new_u, res

    def forward(self, inputs):
        """ Link forward

        :param inputs:
        :return:
        """
        with chainer.no_backprop_mode():
            x_init, C_hat, c_hat, F_hat, f_hat = inputs
            self.retain_inputs((0, 1, 2, 3, 4))
            if self.no_op_forward:
                self.retain_outputs((0, 1))
                return self.current_states, self.controls
            x_init = to_xp(x_init)
            C_hat = to_xp(C_hat)
            c_hat = to_xp(c_hat)
            F_hat = to_xp(F_hat)
            f_hat = to_xp(f_hat)
            if self.need_expand is True:
                # Taylor expansion
                # grad(delta_x,delta_u) = hessian(0,0) @ (delta_x, delta_u) + grad(0,0)
                c_back = []  # eq(12) in [1], constant term in eq(5.12) can be removed
                for t in range(self.T):
                    xt = self.current_states[t]
                    ut = self.controls[t]
                    xut = self.xp.concatenate((xt, ut), axis=1)
                    assert xut.shape == (self.n_batch, self.n_sc), "expected " + str([self.n_batch, self.n_sc]) + \
                                                                   "acutal" + str(xut.shape)
                    c_back.append(xpbmv(C_hat[t], xut) + c_hat[t])
                c_hat = self.xp.stack(c_back)
                f_hat = None  # eq(13) in [1]
            Ks, ks, _backward = self.backward_rec(C_hat, c_hat, F_hat, f_hat)
            x, u, _forward = self.forward_rec(Ks, ks, self.true_cost, self.true_dynamics, self.ls_decay,
                                              self.max_ls_iter)
        assert list(x.shape) == [self.T, self.n_batch, self.n_state], "x dim mismatch"
        self.back_out = _backward
        self.for_out = _forward
        self.retain_outputs((0, 1))
        assert list(x.shape) == [self.T, self.n_batch, self.n_state]
        assert list(u.shape) == [self.T, self.n_batch, self.n_ctrl]
        assert not self.xp.isnan(u).any()
        return x, u

    def backward(self, target_input_indexes, grad_outputs):
        """

        :param target_input_indexes:
        :param grad_outputs:
        :return:
        """
        # print("backward is called")
        x_init, C_hat, c_hat, F_hat, f_hat = self.get_retained_inputs()
        dl_dx, dl_du = grad_outputs

        x_init = to_xp(x_init)
        C_hat = to_xp(C_hat)
        c_hat = to_xp(c_hat)
        F_hat = to_xp(F_hat)
        f_hat = to_xp(f_hat)
        dl_dx = to_xp(dl_dx)
        dl_du = to_xp(dl_du)
        if dl_dx is None:
            dl_dx = self.xp.zeros((self.T, self.n_batch, self.n_state))
        else:
            assert list(dl_dx.shape) == [self.T, self.n_batch, self.n_state]
        if dl_du is None:
            dl_du = self.xp.zeros((self.T, self.n_batch, self.n_ctrl))
        else:
            assert list(dl_du.shape) == [self.T, self.n_batch, self.n_ctrl]
        # just concatenating dl_dx, dl_du
        d_taus = self.xp.concatenate((dl_dx, dl_du), axis=2)
        # assert False, str(d_taus)
        # choose active control
        new_x, new_u = self.get_retained_outputs()
        new_x = to_xp(new_x)
        new_u = to_xp(new_u)
        active_index = (self.xp.absolute(new_u - self.u_lower) <= 1e-8) | \
                       (self.xp.absolute(new_u - self.u_upper) <= 1e-8)
        dx_init_zero = self.xp.zeros_like(x_init)
        # backward pass LINE (1)
        '''
        print("I", active_index)
        print("r",d_taus)
        print("F",F_hat)
        print("C",C_hat)
        assert  False
        '''
        lqr = LQR_active(dx_init_zero, C_hat, -d_taus, F_hat, None, self.T, self.n_state,
                         self.n_ctrl, u_zero_Index=active_index)
        dx, du = lqr.solve_recursion()
        # print("dx", dx)
        # print("du", du)
        # assert  False
        dxu = self.xp.concatenate((dx, du), axis=2)

        xu = self.xp.concatenate((new_x, new_u), axis=2)
        dC = self.xp.zeros_like(C_hat)
        for t in range(self.T):
            xut = self.xp.concatenate((new_x[t], new_u[t]), axis=1)
            dxut = dxu[t]
            dCt = -0.5 * (xpbger(dxut, xut) + xpbger(xut, dxut))
            assert dC[t].shape == dCt.shape
            dC[t] = dCt
        dc = -dxu
        # Compute Lambda (Forward Pass line(2)) in Module(1)
        # lams = []
        lams = self.xp.zeros((self.T, self.n_batch, self.n_state))
        prev_lam = None
        for t in range(self.T - 1, -1, -1):
            Ct_xx = C_hat[t, :, :self.n_state, :self.n_state]
            Ct_xu = C_hat[t, :, :self.n_state, self.n_state:]
            ct_x = c_hat[t, :, :self.n_state]
            xt = new_x[t]
            ut = new_u[t]
            lamt = xpbmv(Ct_xx, xt) + xpbmv(Ct_xu, ut) + ct_x
            if prev_lam is not None:
                Fxt = self.xp.transpose(F_hat[t, :, :, :self.n_state], axes=(0, 2, 1))
                lamt += xpbmv(Fxt, prev_lam)
            lams[t] = lamt
            prev_lam = lamt
        # lams = list(reversed(lams))
        # Backward Pass Line(3)
        # Compute the derivatives
        # d_Lambda
        # dlams = []
        dlams = self.xp.zeros_like(lams)
        prev_dlam = None
        for t in range(self.T - 1, -1, -1):
            dCt_xx = C_hat[t, :, :self.n_state, :self.n_state]
            dCt_xu = C_hat[t, :, :self.n_state, self.n_state:]
            drt_x = -d_taus[t, :, :self.n_state]
            dxt = dx[t]
            dut = du[t]
            dlamt = xpbmv(dCt_xx, dxt) + xpbmv(dCt_xu, dut) + drt_x
            if prev_dlam is not None:
                Fxt = self.xp.transpose(F_hat[t, :, :, :self.n_state], axes=(0, 2, 1))
                dlamt += xpbmv(Fxt, prev_dlam)
            dlams[t] = dlamt
            prev_dlam = dlamt
        # dlams = self.xp.stack(list(reversed(dlams)))
        # d_F
        dF = self.xp.zeros_like(F_hat)
        for t in range(self.T - 1):
            xut = xu[t]
            lamt = lams[t + 1]
            dxut = dxu[t]
            dlamt = dlams[t + 1]
            append_to_dF = -(xpbger(dlamt, xut) + xpbger(lamt, dxut))
            assert dF[t].shape == append_to_dF.shape, str(dF[t].shape) + " : " + str(append_to_dF.shape)
            dF[t] = append_to_dF
        if f_hat is not None:
            _dlams = dlams[1:]
            assert _dlams.shape == f_hat.shape
            df = -_dlams
            df = chainer.Variable(df)
        else:
            # CHECK THIS
            df = chainer.Variable()

        dx_init = -dlams[0]

        # print(dx_init.shape)
        # print(dC.shape)
        # print(dc.shape)
        # print(dF.shape)
        # print(dF)
        # assert False
        dx_init = chainer.Variable(dx_init)
        dC = chainer.Variable(dC)
        dc = chainer.Variable(dc)
        dF = chainer.Variable(dF)
        # print("dC", dC)
        # print("dc", dc)
        return dx_init, dC, dc, dF, df
