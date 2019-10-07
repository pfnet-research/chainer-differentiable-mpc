#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Taylor approximation approximation
"""
import pathlib
import sys

import chainer
from chainer import functions as F

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
from util import bmv, to_xp


def approximate_cost(x, u, Cf):
    """ approximate cost function at point(x, u)

    :param x: time batch n_state
    :param u: time batch n_ctrl
    :param Cf:Cost Function need map vector to scalar
    :return: hessian, grads, costs
    """
    assert x.shape[0] == u.shape[0]
    assert x.shape[1] == u.shape[1]
    T = x.shape[0]
    tau = F.concat((x, u), axis=2)
    costs = []
    hessians = []
    grads = []
    # for time
    for t in range(T):
        tau_t = tau[t]
        cost = Cf(tau_t)  # value of cost function at tau
        assert list(cost.shape) == [x.shape[1]]
        # print("cost.shape", cost.shape)
        grad = chainer.grad([F.sum(cost)], [tau_t], enable_double_backprop=True)[0]  # need hessian
        hessian = []
        # for each dimension?
        for v_i in range(tau.shape[2]):
            # n_sc
            grad_line = F.sum(grad[:, v_i])
            hessian.append(chainer.grad([grad_line], [tau_t])[0])
        hessian = F.stack(hessian, axis=-1)
        costs.append(cost)
        # change to near 0?? Is this necessary ???
        grads.append(grad - bmv(hessian, tau_t))
        hessians.append(hessian)
    costs = F.stack(costs)
    grads = F.stack(grads)
    hessians = F.stack(hessians)
    return hessians, grads, costs


def test_cost():
    import numpy as np
    np.random.seed(0)
    batch = 2
    time = 3
    n_state = 1
    n_ctrl = 1
    x = np.random.randn(time, batch, n_state)
    u = np.random.randn(time, batch, n_ctrl)
    x = chainer.Variable(x)
    u = chainer.Variable(u)
    Cf = lambda invari: F.sqrt(F.sum(invari ** 2, axis=1))
    hessian, grad, cost = approximate_cost(x, u, Cf)
    print("grad x", x / F.expand_dims(cost, axis=2))
    print("grad u", u / F.expand_dims(cost, axis=2))
    print("hessian", hessian)
    print("grad", grad)
    print("cost", cost)


def linearize_dynamics(x, u, dynamics):
    """linearize dynamics

    :param x:time batch n_state
    :param u:
    :param dynamics:
    :return:
    """
    assert x.shape[0] == u.shape[0]
    assert x.shape[1] == u.shape[1]
    n_state = x.shape[2]
    T = x.shape[0]
    x_init = x[0]
    x_ar = [x_init]
    # NOTE need to use newly calculate trajectory ???
    # TODO: CHECK THIS
    large_F, f = [], []
    for t in range(T):
        if t < T - 1:
            xt = x_ar[t]
            ut = u[t]
            #  print("x_ut.shape", xut.shape)
            new_x = dynamics(xt, ut)
            # Linear dynamics approximation.
            Rt, St = [], []
            for j in range(n_state):
                Rj, Sj = chainer.grad([F.sum(new_x[:, j])], [xt, ut])
                Rt.append(Rj)
                St.append(Sj)
                assert Sj is not None
            Rt = F.stack(Rt, axis=1)
            St = F.stack(St, axis=1)
            #  print("Rt shape", Rt.shape)
            #  print("St shape", St.shape)
            Ft = F.concat((Rt, St), axis=2)
            large_F.append(Ft)
            ft = new_x - bmv(Rt, xt) - bmv(St, ut)
            f.append(ft)
            x_ar.append(new_x)

    large_F = F.stack(large_F, 0)
    f = F.stack(f, 0)
    return large_F, f


def test_dynamics():
    import numpy as np
    np.random.seed(0)
    batch = 2
    time = 3
    n_state = 1
    n_ctrl = 1
    n_sc = n_state + n_ctrl
    x = np.random.randn(time, batch, n_state)
    u = np.random.randn(time, batch, n_ctrl)
    x = chainer.Variable(x)
    u = chainer.Variable(u)
    A = np.random.randn(batch, n_state, n_sc)
    B = np.random.randn(batch, n_state)
    A = chainer.Variable(A)
    B = chainer.Variable(B)
    dynamics = lambda s, c: bmv(A, F.concat((s, c), axis=1)) + B
    large_F, f = linearize_dynamics(x, u, dynamics)
    print("A", A)
    print("large_F", large_F)
    print("B", B)
    print("f", f)


if __name__ == '__main__':
    test_dynamics()
