#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
utility functions
some functions are copied from chainer optnet
"""
import copy
import operator
from collections import namedtuple

import chainer
import numpy as np
import scipy.linalg
import torch
from chainer import functions as F

try:
    import cupy

    cupy_available = True
except ImportError:
    cupy_available = False

QuadCost = namedtuple('QuadCost', 'C c')
# QuadCost has C and c
LinDx = namedtuple('LinDx', 'F f')
# LinDx has F f
# set default
# https://stackoverflow.com/questions/11351032
QuadCost.__new__.__defaults__ = (None,) * len(QuadCost._fields)
LinDx.__new__.__defaults__ = (None,) * len(LinDx._fields)

_seen_tables = []


def chainer_diag(q):
    """ q

    :param q:
    :return:
    """
    dim = q.shape[0]
    xp = get_array_module(q)
    zeros_matrix = xp.zeros((dim, dim))
    condition_matrix = zeros_matrix.astype('bool')
    for i in range(dim):
        condition_matrix[i][i] = True
    diag_mat = F.where(condition_matrix, q, zeros_matrix)
    return diag_mat


def test_chainer_diag():
    q = [1.2, 2, 3]
    q = np.array(q)
    q = chainer.Variable(q)
    print(chainer_diag(q))
    return None


def to_xp(x):
    if type(x) == chainer.Variable:
        return x.array
    return x


def table_log(tag, d):
    # TODO: There's probably a better way to handle formatting here,
    # or a better way altogether to replace this quick hack.
    global _seen_tables

    def print_row(r):
        print('| ' + ' | '.join(r) + ' |')

    if tag not in _seen_tables:
        print_row(map(operator.itemgetter(0), d))
        _seen_tables.append(tag)

    s = []
    for di in d:
        assert len(di) in [2, 3]
        if len(di) == 3:
            e, fmt = di[1:]
            try:
                s.append(fmt.format(e))
            except:
                s.append(fmt.format(e.data))
        else:
            e = di[1]
            s.append(str(e))
    print_row(s)


def get_array_module(a):
    if cupy_available and isinstance(a, cupy.ndarray):
        return cupy
    else:
        return np


def clamp(x, lower, upper):
    """ Naive Clamping in [2] A
    [[ x ]]_b = min(max(u,b_lower),b_upper)
    :param x:
    :param lower:
    :param upper:
    :return:
    """
    # Not None
    assert x.shape == lower.shape
    assert x.shape == upper.shape
    assert (lower.array <= upper.array).all(), " lower is larger than upper" \
                                               + " lower: " + str(lower) + "upper: " + str(upper)
    return F.minimum(F.maximum(x, lower), upper)


def xpclamp(x, lower, upper):
    assert x.shape == lower.shape, str(x.shape) + " : " + str(lower.shape)
    assert x.shape == upper.shape
    xp = get_array_module(x)
    assert (lower <= upper).all()

    return xp.minimum(xp.maximum(x, lower), upper)


def get_cost(T, u, cost, dynamics=None, x_init=None, x=None):
    """ calculate total cost

    :param T:
    :param u:
    :param cost:
    :param dynamics:
    :param x_init: initial state
    :param x: all states (which includes initial state)
    :return:
    """
    assert x_init is not None or x is not None
    C = None
    c = None
    if isinstance(cost, QuadCost):
        C = cost.C
        c = cost.c

    if x is None:
        x = get_traj(T, u, x_init, dynamics)

    objs = []
    for t in range(T):
        xt = x[t]
        ut = u[t]
        xut = F.concat((xt, ut))
        if isinstance(cost, QuadCost):
            obj = 0.5 * bquad(xut, C[t]) + bdot(xut, c[t])
        else:
            obj = cost(xut)
        objs.append(obj)
    objs = F.stack(objs, axis=0)
    total_obj = F.sum(objs, axis=0)
    return total_obj


def xpget_cost(T, u, cost, dynamics=None, x_init=None, x=None):
    """ calculate total cost

    :param T:
    :param u:
    :param cost:
    :param dynamics:
    :param x_init: initial state
    :param x: all states (which includes initial state)
    :return:
    """
    assert x_init is not None or x is not None
    C = None
    c = None
    xp = get_array_module(u)
    if isinstance(cost, QuadCost):
        C = cost.C
        c = cost.c
        C = to_xp(C)
        c = to_xp(c)

    if x is None:
        x = xpget_traj(T, u, x_init, dynamics)

    objs = []
    for t in range(T):
        xt = x[t]
        ut = u[t]
        xut = xp.concatenate((xt, ut), axis=1)
        if isinstance(cost, QuadCost):
            obj = 0.5 * xpbquad(xut, C[t]) + xpbdot(xut, c[t])
        else:
            obj = cost(xut)
        objs.append(obj)
    objs = xp.stack(objs, axis=0)
    total_obj = xp.sum(objs, axis=0)
    return total_obj


def get_traj(T, u, x_init, dynamics):
    """calculate torajectory
    
    :param T: time
    :param u: control sequence
    :param x_init: initial state
    :param dynamics: dynamics sequence of function
    :return: state sequence
    """
    large_f = None
    f = None
    if isinstance(dynamics, LinDx):
        large_f = dynamics.F
        f = dynamics.f
        if f is not None:
            # F : time batch state state+control
            # f : state
            assert f.shape[1:] == large_f.shape[1:3]

    x = [x_init]
    for t in range(T):
        xt = x[t]
        ut = u[t]
        if t < T - 1:
            # Dynamics is linear
            if isinstance(dynamics, LinDx):
                xut = F.concat((xt, ut))
                new_x = bmv(large_f[t], xut)
                if f is not None:
                    new_x += f[t]
            else:
                # Dynamics is not linear
                new_x = dynamics(xt, ut)
            x.append(new_x)
    x = F.stack(x, axis=0)
    return x


def xpget_traj(T, u, x_init, dynamics):
    """calculate torajectory

    :param T: time
    :param u: control sequence
    :param x_init: initial state
    :param dynamics: dynamics sequence of function
    :return: state sequence
    """
    large_f = None
    f = None
    xp = get_array_module(u)
    if isinstance(dynamics, LinDx):
        large_f = dynamics.F
        f = dynamics.f
        f = to_xp(f)
        large_f = to_xp(large_f)
        if f is not None:
            # F : time batch state state+control
            # f : state
            assert f.shape[1:] == large_f.shape[1:3]

    x = [x_init]
    for t in range(T):
        xt = x[t]
        ut = u[t]
        if t < T - 1:
            # Dynamics is linear
            if isinstance(dynamics, LinDx):
                xut = xp.concatenate((xt, ut))
                new_x = xpbmv(large_f[t], xut)
                if f is not None:
                    new_x += f[t]
            else:
                # Dynamics is not linear
                new_x = dynamics(xt, ut)
            x.append(new_x)
    x = xp.stack(x, axis=0)
    return x


def bmv(a, x):
    """ Batch matrix vector matmul

    :param a: batch times n times m
    :param x: batch times m
    :return:
    """
    assert a.shape[0] == x.shape[0], "batch mismatch"
    assert a.shape[2] == x.shape[1], "mat mul dim mismatch"
    assert len(x.shape) == 2, " x is not batch vector"
    return F.squeeze(F.matmul(a, F.expand_dims(x, axis=2)), axis=2)


def xpbmv(a, x):
    assert a.shape[0] == x.shape[0], "batch mismatch" + str(a.shape) + "," + str(x.shape)
    assert a.shape[2] == x.shape[1], "mat mul dim mismatch"
    assert len(x.shape) == 2, " x is not batch vector"
    xp = get_array_module(x)
    return xp.squeeze(xp.matmul(a, xp.expand_dims(x, axis=2)), axis=2)


def bger(x, y):
    """ Batch outer product

    :param x:
    :param y:
    :return:
    """
    if x.dtype == 'int' and y.dtype == 'int':
        x_float = F.cast(x, 'float32')
        y_float = F.cast(y, 'float32')
        res_float = F.expand_dims(x_float, 2) @ F.expand_dims(y_float, 1)
        return F.cast(res_float, 'int')
    return F.expand_dims(x, 2) @ F.expand_dims(y, 1)


def xpbger(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    xp = get_array_module(x)
    if x.dtype == 'int' and y.dtype == 'int':
        x_float = xp.cast(x, 'float32')
        y_float = xp.cast(y, 'float32')
        res_float = xp.expand_dims(x_float, 2) @ xp.expand_dims(y_float, 1)
        return xp.cast(res_float, 'int')
    return xp.expand_dims(x, 2) @ xp.expand_dims(y, 1)


def bquad(x, Q):
    """ calcuate x^T Q x

    :param x: vector batch times n
    :param Q: batch times n times n
    :return: batch dim
    """
    assert x.shape[0] == Q.shape[0], "batch mismatch" + str(x.shape) + ":" + str(Q.shape)
    assert x.shape[1] == Q.shape[1], "mat mul dim mismatch"
    assert Q.shape[2] == Q.shape[1], "Q is not square matrix"
    xT = F.expand_dims(x, 1)
    x_ = F.expand_dims(x, 2)
    res = F.squeeze(F.squeeze(xT @ Q @ x_, axis=1), axis=1)
    assert list(res.shape) == [list(x.shape)[0]]
    return res


def xpbquad(x, Q):
    assert x.shape[0] == Q.shape[0], "batch mismatch" + str(x.shape) + ":" + str(Q.shape)
    assert x.shape[1] == Q.shape[1], "mat mul dim mismatch"
    assert Q.shape[2] == Q.shape[1], "Q is not square matrix"
    xp = get_array_module(x)
    xT = xp.expand_dims(x, 1)
    x_ = xp.expand_dims(x, 2)
    res = xp.squeeze(xp.squeeze(xT @ Q @ x_, axis=1), axis=1)
    assert list(res.shape) == [list(x.shape)[0]]
    return res


def expand_time_batch(m, time, n_batch):
    """ add [time, n_batch] dimension
    
    :param m: input chainer variable
    :param time: 
    :param n_batch: 
    :return: time times batch times m
    """
    # expand two dimensions
    m = F.expand_dims(F.expand_dims(m, 0), 0)
    # repeat  along batch dimension
    m = F.repeat(m, n_batch, axis=1)
    # repeat along time dimension
    m = F.repeat(m, time, axis=0)
    assert list(m.shape)[0] == time
    assert list(m.shape)[1] == n_batch
    return m


def expand_batch(m, n_batch):
    """ add [n_batch] dimension

    :param m: input chainer variable
    :param n_batch:
    :return: batch times m
    """
    # expand two dimensions
    m = F.expand_dims(m, 0)
    # repeat  along batch dimension
    m = F.repeat(m, n_batch, axis=0)
    assert list(m.shape)[0] == n_batch
    return m


def xpexpand_batch(m, n_batch):
    """ add [n_batch] dimension

    :param m: input chainer variable
    :param n_batch:
    :return: batch times m
    """
    xp = get_array_module(m)
    # expand two dimensions
    m = xp.expand_dims(m, 0)
    # repeat  along batch dimension
    m = xp.repeat(m, n_batch, axis=0)
    assert list(m.shape)[0] == n_batch
    return m


def bdot(x, y):
    """ batch direct product
    Not to be confused with Exterior product.

    :param x: batch times n
    :param y: batch times n
    :return: batch dimension
    """
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    xT = F.expand_dims(x, 1)
    y = F.expand_dims(y, 2)
    res = F.squeeze(F.squeeze(xT @ y, axis=1), axis=1)
    return res


def xpbdot(x, y):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    xp = get_array_module(x)
    xT = xp.expand_dims(x, 1)
    y = xp.expand_dims(y, 2)
    res = xp.squeeze(xp.squeeze(xT @ y, axis=1), axis=1)
    return res


def batch_lu_factor(A):
    """ lu factorization

    :param A:
    :return:
    """
    assert len(A.shape) == 3, "Actual" + str(A.shape)
    assert A.shape[1] == A.shape[2], "Actual" + str(A.shape)
    xp = chainer.backend.get_array_module(A)
    A = copy.deepcopy(A)
    if type(A) != xp.ndarray:
        A = A.array
    if cupy_available and xp == cupy:
        Ps = xp.empty((A.shape[0], A.shape[1]), dtype=np.int32)
        for i in range(len(A)):
            A[i], Ps[i] = cupyx.scipy.linalg.u_factor(A[i], overwrite_a=True)
        return A, Ps
    else:
        Ps = []
        for i in range(len(A)):
            A[i], piv = scipy.linalg.lu_factor(A[i], overwrite_a=True)
            Ps.append(piv)
        return chainer.Variable(A), xp.array(Ps)


def xpbatch_lu_factor(A):
    assert len(A.shape) == 3, "Actual" + str(A.shape)
    assert A.shape[1] == A.shape[2], "Actual" + str(A.shape)
    xp = get_array_module(A)
    A = copy.deepcopy(A)
    '''
    if cupy_available and xp == cupy:
        Ps = xp.empty((A.shape[0], A.shape[1]), dtype=np.int32)
        for i in range(len(A)):
            A[i], Ps[i] = cupyx.scipy.linalg.u_factor(A[i], overwrite_a=True)
        return A, Ps
    else:
        Ps = []
        for i in range(len(A)):
            A[i], piv = scipy.linalg.lu_factor(A[i], overwrite_a=True)
            Ps.append(piv)
        return A, Ps
    '''
    # use PyTorch here, because scipy does not offer batch lu_factorization
    A_LU, pivots = torch.lu(torch.tensor(A))
    return A_LU.cpu().numpy(), pivots.cpu().numpy()


def batch_lu_solve(lu_and_piv, b):
    """ solve ax = b

    :param lu_and_piv:
    :param b:
    :return:
    """
    LU, piv = lu_and_piv
    LU = LU.array
    xp = chainer.backend.get_array_module(LU)
    b = b.array
    b = b.copy()
    for i in range(len(LU)):
        if cupy_available and xp == cupy:
            b[i] = cupyx.scipy.linalg.lu_solve((LU[i], piv[i]), b[i], overwrite_b=True)
        else:
            b[i] = scipy.linalg.lu_solve((LU[i], piv[i]), b[i], overwrite_b=True)
    return chainer.Variable(b)


def xpbatch_lu_solve(lu_and_piv, b):
    """ solve ax = b

    :param lu_and_piv:
    :param b:
    :return:
    """
    LU, piv = lu_and_piv
    # xp = get_array_module(LU)
    b = b.copy()
    '''
    for i in range(len(LU)):
        if cupy_available and xp == cupy:
            b[i] = cupyx.scipy.linalg.lu_solve((LU[i], piv[i]), b[i], overwrite_b=True)
        else:
            b[i] = scipy.linalg.lu_solve((LU[i], piv[i]), b[i], overwrite_b=True)
    '''
    b = torch.Tensor(b)
    b = b.float()
    LU = torch.from_numpy(LU)
    piv = torch.from_numpy(piv)
    LU = LU.float()
    b = torch.lu_solve(b, LU, piv)
    return b.cpu().numpy()


if __name__ == '__main__':
    test_chainer_diag()
