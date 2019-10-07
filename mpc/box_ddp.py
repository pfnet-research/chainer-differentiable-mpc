#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Box Differential Dynamic Programming [2]
BAD nameing this seemes to be box constrained iLQR
"""
import copy
import pathlib
import sys
import warnings

import chainer
from approximate import approximate_cost, linearize_dynamics
from chainer import functions as F
from mpc_step import MPCstep

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
sys.path.append(str(current_dir) + '/../lqr')
from util import bmv, get_cost, get_traj, LinDx, QuadCost, table_log, to_xp


class BoxDDP(chainer.Link):
    """BoxDDP solve"""

    def __init__(self, T, u_lower, u_upper, n_batch, n_state, n_ctrl, u_init, eps=1e-5, not_improved_lim=5,
                 line_search_decay=0.2, max_line_search_iter=10, best_cost_eps=1e-4, max_iter=10,
                 detach_unconverged=True, exit_unconverged=True, verbose=False, ilqr_verbose=False,
                 update_dynamics=True):
        """ constructor

        :param T:
        :param u_lower:
        :param u_upper:
        :param n_batch:
        :param n_state:
        :param n_ctrl:
        :param eps: Termination threshold, on the norm of the full control
        step (without line search)
        :param  not_improved_lim: The number of iterations to allow that don't
            improve the objective before returning early.
        :param line_search_decay:
        :param max_line_search_iter:
        :param max_iter: maximum iteration in forward
        :param verbose:
        :param ilqr_verbose:
        """
        super().__init__()
        self.u_lower = u_lower
        self.u_upper = u_upper
        self.T = T
        self.n_batch = n_batch
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_sc = self.n_state + self.n_ctrl
        self.eps = eps
        self.not_improved_lim = not_improved_lim
        self.ls_decay = line_search_decay
        self.max_ls_iter = max_line_search_iter
        self.best_cost_eps = best_cost_eps
        self.max_iter = max_iter
        self.verbose = verbose
        self.ilqr_verbose = ilqr_verbose
        self.u_init = u_init
        self.detach_unconverged = detach_unconverged
        self.exit_unconverged = exit_unconverged
        if type(self.u_lower) is not float:
            assert list(self.u_lower.shape) == [self.T, self.n_batch, self.n_ctrl], 'actual' + str(self.u_lower.shape)
            assert list(self.u_upper.shape) == [self.T, self.n_batch, self.n_ctrl]
        else:
            # u_lower
            self.u_lower = self.xp.array(self.u_lower)
            self.u_lower = self.xp.expand_dims(self.u_lower, 0)
            self.u_lower = self.xp.expand_dims(self.u_lower, 0)
            self.u_lower = self.xp.expand_dims(self.u_lower, 0)
            self.u_lower = self.xp.repeat(self.u_lower, self.n_ctrl, axis=2)
            self.u_lower = self.xp.repeat(self.u_lower, self.T, axis=0)
            self.u_lower = self.xp.repeat(self.u_lower, self.n_batch, axis=1)
            assert list(self.u_lower.shape) == [self.T, self.n_batch, self.n_ctrl], 'expected' + str(
                [self.T, self.n_batch, self.n_ctrl]) + 'actual' + str(self.u_lower.shape)
            # u_upper
            self.u_upper = self.xp.array(self.u_upper)
            self.u_upper = self.xp.expand_dims(self.u_upper, 0)
            self.u_upper = self.xp.expand_dims(self.u_upper, 0)
            self.u_upper = self.xp.expand_dims(self.u_upper, 0)
            self.u_upper = self.xp.repeat(self.u_upper, self.n_ctrl, axis=2)
            self.u_upper = self.xp.repeat(self.u_upper, self.T, axis=0)
            self.u_upper = self.xp.repeat(self.u_upper, self.n_batch, axis=1)
            assert list(self.u_upper.shape) == [self.T, self.n_batch, self.n_ctrl]
        self.update_dynamics = update_dynamics

    def forward(self, inputs):
        """

        :param x_init:
        :param cost: true cost
        :param dynamics: true dynamics
        :return:
        """
        x_init, cost, dynamics = inputs
        assert list(x_init.shape) == [self.n_batch, self.n_state], " x_init dim mismatch"
        if self.u_init is None:
            u = chainer.Variable(self.xp.zeros((self.T, self.n_batch, self.n_ctrl), dtype=x_init.dtype))
        else:
            u = self.u_init
            if list(u.shape) == [self.T, self.n_ctrl]:  # time times n_ctrl
                u_ = copy.deepcopy(u)
                u = F.repeat(F.expand_dims(u_, 1), self.n_batch, axis=1)
            elif type(u) != chainer.Variable:
                u = chainer.Variable(u)
        assert list(u.shape) == [self.T, self.n_batch, self.n_ctrl], "u dim mismatch, actual" + str(u.shape)
        assert not self.xp.isnan(u.array).any()
        if self.verbose is True:
            print('Initial mean(cost): {:.4e}'.format(
                F.mean(get_cost(self.T, u, cost, dynamics, x_init=x_init)).array
            ))
        best = None
        n_not_improved = 0
        assert not self.xp.isnan(u.array).any()
        for i in range(self.max_iter):
            assert not self.xp.isnan(to_xp(u)).any()
            x = get_traj(self.T, u, x_init=x_init, dynamics=dynamics)
            assert not self.xp.isnan(to_xp(u)).any()
            assert list(x.shape) == [self.T, self.n_batch, self.n_state], "x dim mismatch"
            assert list(u.shape) == [self.T, self.n_batch, self.n_ctrl], "u dim mismatch"
            if isinstance(dynamics, LinDx):
                # dynamics is linear no need to be linearized
                large_f, f = dynamics.F, dynamics.f
            else:
                large_f, f = linearize_dynamics(x, u, dynamics)

            if isinstance(cost, QuadCost):
                C, c = cost.C, cost.c
            else:
                C, c, _ = approximate_cost(x, u, cost)
            assert list(large_f.shape) == [self.T, self.n_batch, self.n_state, self.n_sc] or \
                   list(large_f.shape) == [self.T - 1, self.n_batch, self.n_state, self.n_sc], "actual " + str(
                large_f.shape)
            assert list(C.shape) == [self.T, self.n_batch, self.n_sc, self.n_sc], "actual" + str(C.shape)
            if f is not None:
                assert list(f.shape) == [self.T, self.n_batch, self.n_state] or list(f.shape) == [self.T - 1,
                                                                                                  self.n_batch,
                                                                                                  self.n_state]
            assert list(c.shape) == [self.T, self.n_batch, self.n_sc], "expected:" + str(
                [self.T, self.n_batch, self.n_sc]) + "actual:" + str(c.shape)
            '''
            print("x", x)
            print("u", u)
            print("C", C)
            print("c", c)
            print("large_f", large_f)
            print("f", f)
            print("x", x)
            print("cost",cost)
            print("dynamics", dynamics)
            breakpoint()
            '''
            assert not self.xp.isnan(u.array).any()
            mpc_solver = MPCstep(controls=u, T=self.T, u_upper=self.u_upper, u_lower=self.u_lower, n_batch=self.n_batch,
                                 n_state=self.n_state, n_ctrl=self.n_ctrl, current_states=x, true_cost=cost,
                                 true_dynamics=dynamics, ls_decay=self.ls_decay, max_ls_iter=self.max_ls_iter,
                                 verbose=self.ilqr_verbose, need_expand=True)
            _x_init = x[0].array
            if self.update_dynamics is True:
                C = C.array
                c = c.array
            else:
                # print("updating cost function")
                large_f = large_f.array
                f = f.array
            x, u = mpc_solver.apply((_x_init, C, c, large_f, f))
            assert not self.xp.isnan(u.array).any()
            assert not self.xp.isnan(u.array).any()
            assert list(x.shape) == [self.T, self.n_batch, self.n_state], "actual:" + str(x.shape)
            assert list(u.shape) == [self.T, self.n_batch, self.n_ctrl], "actual:" + str(u.shape)
            back_out = mpc_solver.back_out
            for_out = mpc_solver.for_out
            '''
            print("x", x)
            print("u", u)
            print("full_du_norm", for_out.full_du_norm)
            '''
            n_not_improved += 1
            assert len(x.shape) == 3, "x shape mismatch"
            assert len(u.shape) == 3, "u shape mismatch"
            # below is just for monitoring
            if best is None:
                best = {
                    'x': list(F.split_axis(x, indices_or_sections=self.n_batch, axis=1)),
                    'u': list(F.split_axis(u, indices_or_sections=self.n_batch, axis=1)),
                    'costs': for_out.costs,
                    'full_du_norm': for_out.full_du_norm,
                }
                assert list(best['x'][0].shape) == [self.T, 1, self.n_state], \
                    'actual:' + str(best['x'][0].shape)
                assert list(best['u'][0].shape) == [self.T, 1, self.n_ctrl], \
                    'actual:' + str(best['u'][0].shape)
            else:
                for j in range(self.n_batch):
                    if for_out.costs[j] <= best['costs'][j] + self.best_cost_eps:
                        n_not_improved = 0
                        best['x'][j] = F.expand_dims(x[:, j], axis=1)
                        assert list(best['x'][j].shape) == [self.T, 1, self.n_state], \
                            'actual:' + str(best['x'][j].shape)
                        best['u'][j] = F.expand_dims(u[:, j], axis=1)
                        assert list(best['u'][j].shape) == [self.T, 1, self.n_ctrl]
                        best['costs'][j] = for_out.costs[j]
                        best['full_du_norm'][j] = for_out.full_du_norm[j]
            if self.verbose is True:
                #  print("for_out", for_out)
                table_log('lqr', (
                    ('iter', i),
                    ('mean(cost)', self.xp.mean(best['costs']), '{:.4e}'),
                    ('||full_du||_max', self.xp.max(for_out.full_du_norm), '{:.2e}'),
                    # ('||alpha_du||_max', max(for_out.alpha_du_norm), '{:.2e}'),
                    # TODO: alphas, total_qp_iters here is for the current
                    # iterate, not the best
                    ('mean(alphas)', for_out.mean_alphas, '{:.2e}'),
                    ('total_qp_iters', back_out.n_total_qp_iter),
                ))
            # Convergence
            if max(for_out.full_du_norm) < self.eps:
                print("Converged")
                break
            if n_not_improved > self.not_improved_lim:
                print("Not improved lim")
                break
            if i == self.max_iter - 1:
                print("Not Converged ")
        x = F.concat(best['x'], axis=1)
        u = F.concat(best['u'], axis=1)
        full_du_norm = best['full_du_norm']
        # need taylor at new point
        if isinstance(dynamics, LinDx):
            large_f, f = dynamics.F, dynamics.f
        else:
            large_f, f = linearize_dynamics(x, u, dynamics)
        if isinstance(cost, QuadCost):
            C, c = cost.C, cost.c
        else:
            C, c, _ = approximate_cost(x, u, cost)
        costs = best['costs']
        assert list(x.shape) == [self.T, self.n_batch, self.n_state], "actual:" + str(x.shape)
        assert list(u.shape) == [self.T, self.n_batch, self.n_ctrl], "actual:" + str(u.shape)
        # DOES NO OPERATION FORWARD MEAN SENSE?
        mpc_solver = MPCstep(controls=u, T=self.T, u_upper=self.u_upper, u_lower=self.u_lower, n_batch=self.n_batch,
                             n_state=self.n_state, n_ctrl=self.n_ctrl, current_states=x, true_cost=cost,
                             true_dynamics=dynamics, ls_decay=self.ls_decay, max_ls_iter=self.max_ls_iter,
                             verbose=self.ilqr_verbose, need_expand=True, no_op_forward=True)
        _x_init = to_xp(x[0])
        if self.update_dynamics is True:
            C = C.array
            c = c.array
        else:
            # print("updating cost function")
            large_f = large_f.array
            f = f.array
        x_new, u_new = mpc_solver.apply((_x_init, C, c, large_f, f))
        x = x_new
        u = u_new
        # NOTE: codes under this line not checked
        if self.detach_unconverged:
            # DON'T BACKPROPAGATE throught unconverged batch
            if max(best['full_du_norm']) > self.eps:
                if self.exit_unconverged:
                    pass
                    # assert False

                if self.verbose is True:
                    print("LQR Warning: All examples did not converge to a fixed point.")
                    print("Detaching and *not* backpropping through the bad examples.")
                warnings.warn("LQR Warning: All examples did not converge to a fixed point.")
                Index = for_out.full_du_norm < self.eps
                Index_expand = F.expand_dims(F.expand_dims(Index, 0), 2)
                assert len(x.shape) == len(Index_expand.shape)
                Index_x = F.repeat(Index_expand, x.shape[0], axis=0)
                Index_x = F.repeat(Index_x, x.shape[2], axis=2)
                assert x.shape == Index_x.shape, str(x.shape) + ":" + str(Index_x.shape)
                Index_u = F.repeat(Index_expand, u.shape[0], axis=0)
                Index_u = F.repeat(Index_u, u.shape[2], axis=2)
                assert u.shape == Index_u.shape
                Index_x = F.cast(Index_x, 'float')
                Index_u = F.cast(Index_u, 'float')
                # call array to detach. Is this right approach
                _x = copy.deepcopy(x).array
                _u = copy.deepcopy(u).array
                x = x * Index_x + _x * (1. - Index_x)
                u = u * Index_u + _u * (1. - Index_u)

        return x, u, costs
