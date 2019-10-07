# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
print(sys.path)
sys.path.append("../mpc")
from pnqp import PNQP
import chainer
import numpy as np
import matplotlib.pyplot as plt

H = np.array([[[ 7.9325,  4.9520,  1.0314,  0.2282],
         [ 4.9520,  8.7746,  1.7916,  3.3622],
         [ 1.0314,  1.7916,  4.2824, -2.5979],
         [ 0.2282,  3.3622, -2.5979,  6.7064]],

        [[ 3.4423, -1.9137, -0.9978, -4.4905],
         [-1.9137,  6.7254,  3.3720,  1.7444],
         [-0.9978,  3.3720,  3.5695, -0.9766],
         [-4.4905,  1.7444, -0.9766, 13.0806]]])
H = chainer.Variable(H)

q = np.array([[ -0.8277,   8.5116, -12.1597,  17.9497],
        [ -3.5764,  -5.3455,  -3.2465,   4.3960]])
q = chainer.Variable(q)

lower = np.array([[-0.2843, -0.0063, -0.1808, -0.6669],
        [-0.1359, -0.3629, -0.2125, -0.0121]])
lower = chainer.Variable(lower)

upper = np.array([[0.1345, 0.0307, 0.0277, 0.9418],
        [0.6205, 0.2703, 0.4023, 0.2560]])
upper = chainer.Variable(upper)

x_init = None

solve = PNQP(H,q,lower,upper)

# + {"active": ""}
# # tensor([[[ 7.9325,  4.9520,  1.0314,  0.2282],
#          [ 4.9520,  8.7746,  1.7916,  3.3622],
#          [ 1.0314,  1.7916,  4.2824, -2.5979],
#          [ 0.2282,  3.3622, -2.5979,  6.7064]],
#
#         [[ 3.4423, -1.9137, -0.9978, -4.4905],
#          [-1.9137,  6.7254,  3.3720,  1.7444],
#          [-0.9978,  3.3720,  3.5695, -0.9766],
#          [-4.4905,  1.7444, -0.9766, 13.0806]]])
# tensor([[ -0.8277,   8.5116, -12.1597,  17.9497],
#         [ -3.5764,  -5.3455,  -3.2465,   4.3960]])
# tensor([[-0.2843, -0.0063, -0.1808, -0.6669],
#         [-0.1359, -0.3629, -0.2125, -0.0121]])
# tensor([[0.1345, 0.0307, 0.0277, 0.9418],
#         [0.6205, 0.2703, 0.4023, 0.2560]])
# None
# tensor([[ 0.1239, -0.0063,  0.0277, -0.6669],
#         [ 0.6205,  0.2703,  0.4023, -0.0121]])
