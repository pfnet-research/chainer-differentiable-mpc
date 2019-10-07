#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Pendulum dynamics
Environment file
"""
import os
import pathlib
import sys
import tempfile

import chainer
# import cupy as xp
import matplotlib
import numpy as xp
from chainer import Variable
from chainer import functions as F

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')

from util import to_xp

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')


class PendulumDx(chainer.Link):
    def __init__(self, params=None, simple=True):
        """

        :param params:
        :param simple: not need dumping and gravity bias
        """
        super().__init__()
        self.simple = simple
        self.max_torque = 2.0
        self.dt = 0.05
        self.n_state = 3
        self.n_ctrl = 1
        if params is None:
            if simple:
                # gravity(g), mass(m), length(l)
                self.params = Variable(xp.array([10., 1., 1.]))
            else:
                # gravity (g), mass (m), length (l), damping (d), gravity bias (b)
                self.params = Variable(xp.array([10., 1., 1., 0., 0.]))
        else:
            self.params = params

        assert len(self.params) == 3 if simple else 5

        self.goal_state = xp.array([1., 0., 0.])
        self.goal_weights = xp.array([1., 1., 0.1])
        self.ctrl_penalty = 0.001
        self.lower = -2.
        self.upper = 2.
        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        # x, u = inputs
        squeeze = len(x.shape) == 1
        if squeeze:
            # print("squeeze is true")
            x = F.expand_dims(x, 0)
            u = F.expand_dims(u, 0)
        assert len(x.shape) == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == self.n_state, str(x.shape[1])
        assert u.shape[1] == self.n_ctrl
        # x (n_batch, n_state)
        # u (n_batch, n_ctrl)
        assert len(u.shape) == 2, str(u.shape)
        if not hasattr(self, 'simple') or self.simple:
            g, m, l = F.separate(self.params)
        else:
            g, m, l, d, b = F.separate(self.params)

        u = F.clip(u, -self.max_torque, self.max_torque)[:, 0]
        # cos, sin, angular velocity
        cos_th, sin_th, dth = F.separate(x, axis=1)
        # theta
        th = F.arctan2(sin_th, cos_th)
        # calculate new angular velocity
        if not hasattr(self, 'simple') or self.simple:
            newdth = dth
            newdth += self.dt * (-3. * g / (2. * l) * (-sin_th) + 3. * u / (m * l ** 2))
        else:
            sin_th_bias = F.sin(th + b)
            newdth = dth
            newdth += self.dt * (-3. * g / (2. * l) * (-sin_th_bias) + 3. * u / (m * l ** 2) - d * th)
        newth = th + newdth * self.dt
        state = F.stack((F.cos(newth), F.sin(newth), newdth), axis=1)

        if squeeze:
            state = F.squeeze(state, axis=0)
        return state

    def get_frame(self, x, ax=None):
        x = to_xp(x)
        assert len(x) == 3, str(len(x)) + ":" + str(x)
        l = self.params[2].array
        cos_th, sin_th, dth = x
        th = xp.arctan2(sin_th, cos_th)
        x = sin_th * l
        y = cos_th * l
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        ax.plot((0, x), (0, y), color='k')
        ax.set_xlim((-l * 1.2, l * 1.2))
        ax.set_ylim((-l * 1.2, l * 1.2))
        return fig, ax

    def get_true_obj(self):
        """ get true cost
        q assumes only diagnonal element of Q

        :return:
        """
        q = xp.concatenate((
            self.goal_weights,
            self.ctrl_penalty * xp.ones(self.n_ctrl)
        ))
        '''
        self.goal_state = xp.array([1., 0., 0.])
        self.goal_weights = xp.array([1., 1., 0.1])
        self.ctrl_penalty = 0.001
        '''
        assert not hasattr(self, 'mpc_lin')
        # ctrl penalty is always applied to squared norm
        px = - xp.sqrt(self.goal_weights) * self.goal_state  # + self.mpc_lin
        p = xp.concatenate((px, xp.zeros(self.n_ctrl)))
        '''
        q = array([1.   , 1.   , 0.1  , 0.001])
        p = array([-1., -0., -0.,  0.])
        '''
        return q, p


def make_gif(controls, xinit, prefix=None):
    """

    :param controls:
    :return:
    """
    dx = PendulumDx()
    T = controls.shape[0]
    x = xinit
    t_dir = tempfile.mkdtemp()
    print('Tmp dir: {}'.format(t_dir))
    for t in range(T):
        # print(type(x))
        # print(type(u[t]))
        x = dx(x, controls[t])
        assert len(x[0]) == 3, str(x[0].shape)
        fig, ax = dx.get_frame(x[0])
        fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
        plt.close(fig)
    if prefix is not None:
        vid_file = prefix + 'pendulum_vid.mp4'
    else:
        vid_file = 'pendulum_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)

    cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
        t_dir, vid_file)
    os.system(cmd)
    return vid_file


def make_long_gif(controller, x_init, T, prefix=None):
    """
    :param controler:
    :param x_init:
    :param T:
    :param n_iter:
    :return:
    """
    dx = PendulumDx()
    x = x_init
    t_dir = tempfile.mkdtemp()
    print('Tmp dir: {}'.format(t_dir))
    controls = controller(x)
    for t in range(T):
        # print(type(x))
        # print(type(u[t]))
        x = dx(x, controls[t])
        assert len(x[0]) == 3, str(x[0].shape)
        if t == T - 1:
            pass
        else:
            fig, ax = dx.get_frame(x[0])
            fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
        plt.close(fig)
    controls = controller(x)
    for t in range(T - 1, T - 1 + T):
        x = dx(x, controls[t - T + 1])
        fig, ax = dx.get_frame(x[0])
        fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
        plt.close(fig)
    if prefix is not None:
        vid_file = prefix + 'pendulum_vid.mp4'
    else:
        vid_file = 'pendulum_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)

    cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
        t_dir, vid_file)
    os.system(cmd)
    return vid_file


def test_make_gif():
    from il_env import IL_Env
    n_train, n_val, n_test = 0, 0, 1
    env = IL_Env('pendulum', lqr_iter=500)
    env.populate_data(n_train=n_train, n_val=n_val, n_test=n_test, seed=0)
    print(env.test_data.shape)
    xinit = env.test_data[0][0][:3].reshape(1, 3)
    u = env.test_data[0][:, 3].reshape(env.mpc_T, 1, 1)
    # print("xinit shape", xinit.shape)
    # print("u shape", u.shape)
    # print("x dtype", xinit.dtype)
    # print("u dtype", u.dtype)
    # print("x type", type(xinit))
    # print("u type", type(u))
    res = make_gif(u, xinit)
    # print("res", res)


def test_long_gif():
    from il_env import IL_Env
    x_init = IL_Env.sample_xinit(n_batch=1)
    print(x_init)
    x_init[0][0] = xp.cos(0.1)
    x_init[0][1] = xp.sin(0.1)
    x_init[0][2] = 0.0
    x_init[0, :] = xp.array([[0.97434908, 0.22504192, -0.56898465]])
    env = IL_Env('pendulum', lqr_iter=500)
    true_q, true_p = env.true_dx.get_true_obj()
    controller = lambda _xinit: env.mpc(env.true_dx, _xinit, true_q, true_p, update_dynamics=True)[1]
    make_long_gif(controller, x_init, env.mpc_T)


if __name__ == '__main__':
    # test_make_gif()
    test_long_gif()
