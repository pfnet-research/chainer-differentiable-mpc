#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Box constrained Quadratic Programming
Solve this problem as described in [2]
Projected Newton Quadratic Programming
minimize f(x) = 1/2 x^THx + q^Tx
subject to b_lower <= x <= b_upper
"""
import pathlib
import sys

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
from util import xpbquad, xpbdot, xpbmv, xpbatch_lu_factor, xpbatch_lu_solve, xpexpand_batch, xpclamp, xpbger, \
    get_array_module, to_xp
import warnings
import copy
from chainer import Variable

# Reduction ratio
GAMMA = 0.1


def calc_obj(H, q, x):
    """ calculate objective function
    :param H:
    :param q:
    :param x:
    :return:
    """
    return 0.5 * xpbquad(x, H) + xpbdot(q, x)


#  @profile
def PNQP(H, q, lower, upper, x_init=None, n_iter=20):
    """ projected newton qp solver
    :param H:
    :param q:
    :param lower:
    :param upper:
    :param x_init:
    :param n_iter:
    :return:
    Algorithm[1] in [2]
    1) Get indices: eq(15)
    2) Get Newton step: eq(16)
    3) Convergence: If |g_f|< epsilon << 1 terminate
    4) Line search
    """
    xp = get_array_module(H)
    if type(H) == Variable:
        H = H.array
    if type(q) == Variable:
        q = q.array
    if type(lower) == Variable:
        lower = lower.array
    if type(upper) == Variable:
        upper = upper.array
    if x_init is not None and type(x_init) == Variable:
        x_init = x_init.array
    assert type(H) == xp.ndarray, str(H)
    assert (lower <= upper).all(), " lower is larger than upper" \
                                   + " lower: " + str(lower) + "upper: " + str(upper)
    n_batch = H.shape[0]
    n_dim = H.shape[1]
    assert list(H.shape) == [n_batch, n_dim, n_dim], "H dim mismatch"
    assert list(q.shape) == [n_batch, n_dim], "q dim mismatch expected" + str([n_batch, n_dim])
    assert list(lower.shape) == [n_batch, n_dim], "lower dim mismatch actual" + str(lower.shape)
    assert list(upper.shape) == [n_batch, n_dim], "upper dim mismatch"
    # small identity matrix
    I_pnqp = xpexpand_batch((1e-11 * xp.eye(n_dim)), n_batch).reshape(n_batch, n_dim, n_dim)
    # print("I_pnqp: ", I_pnqp)
    if x_init is None:
        # make initial guess
        if n_dim == 1:
            x_init = -(1.0 / xp.squeeze(H, axis=2)) * q
        else:
            H_lu = xpbatch_lu_factor(H)
            # Clamped in the x assignment
            # Hx = -q (Don't to unpack H_lu)
            x_init = - xpbatch_lu_solve(H_lu, q)
    else:
        # Don't over-write the original x_init.
        x_init = copy.deepcopy(x_init)
        x_init = to_xp(x_init)
        assert type(x_init[0][0]) != Variable
    # Begin with feasible guess
    assert type(x_init[0][0]) != Variable
    assert type(lower) != Variable
    assert type(upper) != Variable
    x = xpclamp(x_init, lower, upper)
    assert list(x.shape) == [n_batch, n_dim], "x dim mismatch"
    for i in range(n_iter):
        # 1. Get indices
        # calculate gradient
        grad = xpbmv(H, x) + q
        assert type(H) == xp.ndarray
        assert type(grad) == xp.ndarray
        assert type(x) == xp.ndarray
        assert type(lower) == xp.ndarray
        assert type(upper) == xp.ndarray
        try:
            xp.greater(grad, 0.0)
        except:
            print(x)
            print(type(grad))
            print(type(H[0, 0]))
        Index_c = ((x == lower) & (xp.greater(grad, 0.0)) | ((x == upper) & (xp.less(grad, 0.0))))
        Index_c = 1.0 * Index_c
        Index_f = 1.0 - Index_c
        # 2. Get Newton step
        # print(Index_f)
        Index_Hff = xpbger(Index_f, Index_f)
        Index_not_Hff = 1.0 - Index_Hff
        Index_fc = xpbger(Index_f, Index_c)
        Index_c = Index_c.astype('bool')
        g_f = copy.deepcopy(grad)
        # print("g_f original", g_f)
        g_f[Index_c] = 0.0
        # g_f = F.where(Index_c, chainer.Variable(xp.zeros_like(g_f, dtype=g_f.dtype)), g_f)
        # Bad implementation (when n_dim is large)
        H_f = copy.deepcopy(H)
        # Index_not_Hff = xp.cast(Index_not_Hff, 'bool')
        Index_not_Hff = Index_not_Hff.astype('bool')
        H_f[Index_not_Hff] = 0.0
        # H_f = F.where(Index_not_Hff, chainer.Variable(xp.zeros_like(H_f, dtype=H_f.dtype)), H_f)
        H_f += I_pnqp
        # print("H", H)
        # print("H_f", H_f)
        # calculate dx
        if n_dim == 1:
            dx = -(1.0 / xp.squeeze(H_f, axis=2)) * g_f
        else:
            H_lu_f = xpbatch_lu_factor(H_f)
            dx = - xpbatch_lu_solve(H_lu_f, g_f)
        # 3. Convergence
        norm = xp.sqrt(xp.sum(dx ** 2, axis=1))
        batch_large = norm >= 1e-4
        batch_large = batch_large.astype('float')
        num_large = xp.sum(batch_large)
        if num_large == 0:
            return x, H_f if n_dim == 1 else H_lu_f, Index_f, i
        # check convergence
        if num_large.data == 0:
            '''
            print("x:", x)
            if n_dim != 1:
                print("==============================")
                print("H", H_f)
                print(len(H_lu_f))
                print("H_lu_f[0][0]", H_lu_f[0][0])
                print("H_lu_f[0][1]", H_lu_f[0][1])
                print("H_lu_f[1]", H_lu_f[1])
                print("dx", dx)
                print("Index_f", Index_f)
                print("i", i)
            '''
            return x, H_f if n_dim == 1 else H_lu_f, Index_f, i
        # 4. Line search (Backtracking)
        alpha = xp.ones(n_batch, dtype=x.dtype)
        DECAY = 0.1  # making alpha smaller
        max_lhs = xp.array(GAMMA)
        batch_large = batch_large.astype('bool')
        '''
        print("Hf:", H_f)
        print("batch_large: ", batch_large)
        print("gradient:", grad)
        '''
        count = 0
        while max_lhs <= GAMMA and count < 10:
            x_hat = xpclamp(x + xp.diagflat(alpha) @ dx, lower, upper)
            lhs = (GAMMA + 1e-6) * (xp.ones(n_batch, dtype=x.dtype))
            lhs[batch_large] = (calc_obj(H, q, x) - calc_obj(H, q, x_hat))[batch_large] \
                               / xpbdot(grad, x - x_hat)[batch_large]
            '''
            print("x:", x)
            print("dx:", dx)
            print("x_hat: ", x_hat)
            print("lhs_cng", lhs_cng)
            print("x_hat:", x_hat)
            print("lhs:", lhs)
            '''
            I = lhs <= GAMMA
            alpha[I] *= DECAY  # making smaller
            max_lhs = xp.max(lhs)
            # Don't write cnt += cnt +1 HERE
            count += 1
        x = x_hat

    warnings.warn("Projected Newton Quadratic Programming warning: Did not converge")
    '''
    x = Variable(x)
    H_f = Variable(H_f)
    a, b = H_lu_f
    a = Variable(a)
    H_lu_f = (a, b)
    Index_f =Variable(Index_f)
    '''
    return x, H_f if n_dim == 1 else H_lu_f, Index_f, i
