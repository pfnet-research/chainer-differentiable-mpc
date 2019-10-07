# -*- coding: utf-8 -*-
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
sys.path.append("../lqr")
from lqr_recursion import LqrRecursion
import chainer
import numpy as np
import matplotlib.pyplot as plt

T =51
f = None
n_state =3
n_ctrl =1
n_sc = n_ctrl +n_state
F =chainer.Variable(np.array([(np.array([[
    1.0,0, 0, 1],
    [1,1.0,0,0],
   [0, 1, 1, 0]])) for i in range(T)])).reshape(T,1,n_state,n_sc,)
c = chainer.Variable(np.array([(np.array([0,0,0.0,0]).T) for i in range(T)])).reshape(T,1,n_sc,)
_C = np.array([np.array([[0,0 ,0,0],[0,0,0,0],[0,0,1.0,0],[0,0,0,1]]) for i in range(T-1)])
_C = np.append(_C , np.array([[0,0 ,0,0],[0,0,0,0],[0,0,1.0,0],[0,0,0,0.00000000000001]]))
C = chainer.Variable(_C).reshape(T,1,n_sc, n_sc)
x_init = chainer.Variable(np.array([0.5428, 0.7633,0.3504])).reshape(1,n_state)

C

test = LqrRecursion(x_init,C,c,F,f,T,n_state,n_ctrl)

Ks, ks = test.backward()

k1 =[]
k2 = []
fig, ax = plt.subplots()
for i in range(T-1):
    k1.append(Ks[i][0][0][0].data)
    k2.append(Ks[i][0][0][1].data)
major_ticks = np.arange(0,T, 2)
ax.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 1)
ax.grid(which = "major", axis = "y", color = "green", alpha = 0.8,
        linestyle = "--", linewidth = 1)
ax.set_xticks(major_ticks)            
ax.set_ylim(-0.5, 1.2)
ax.plot(k1)
ax.plot(k2)
ax.set_ylim(-2, 0)
ax.set_xlim(0,T)

x,u = test.solve_recursion()

# +
us = []
for i in range(T):
    us.append(x[i][0][0].data)
    
fig, ax = plt.subplots()
ax.grid(which = "major", axis = "x", color = "blue", alpha = 0.8,
        linestyle = "--", linewidth = 1)

# y軸に目盛線を設定
ax.grid(which = "major", axis = "y", color = "green", alpha = 0.8,
        linestyle = "--", linewidth = 1)

major_ticks = np.arange(0, 20, 2)                                              
ax.set_xticks(major_ticks)                                                       
ax.set_ylim(-2, 1)
ax.set_xlim(0, 20)
ax.plot(us, marker='.')
plt.show()
# -

Ks

Ks

len(Ks)

x




