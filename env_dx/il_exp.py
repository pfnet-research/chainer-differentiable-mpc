#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Imitation learning experiment

Warm starts means use optimal trajectory claculated before
"""

import os
import pathlib
import pickle as pkl
import shutil
import time

import chainer
import numpy as xp
from chainer import dataset
from chainer import functions as F
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator
from pendulum_net import Pendulum_Net_cost_logit, Pendulum_Net_cost_lower_triangle, OBSERVATION_MATRIX
from pendulum_net import Pendulum_Net_cost_logit_strange_obervation, Pendulum_Net_cost_lower_triangle_strange_obervation

# import chainer.computational_graph as c
'''
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)
'''

random_init = None


class IL_Exp:
    def __init__(self, n_batch, fname, n_epoch=300, cost=True, dx=False, is_lower_triangle=False,
                 is_strange_observation=False):
        self.n_batch = n_batch
        self.n_epoch = n_epoch
        fname += '_epoch:' + str(n_epoch) + "_"
        current_dir = pathlib.Path(__file__).resolve().parent
        path = str(current_dir) + '/data/'
        self.data_path = path + 'pendulum.pkl'
        # setting environment by reading pkl file
        with open(self.data_path, 'rb') as f:
            self.env = pkl.load(f)
        tag = fname
        self.learn_cost = cost
        self.learn_dx = dx
        if self.learn_cost:
            tag += '.learn_cost'
        if self.learn_dx:
            tag += '.learn_dx'
        # setproctitle('imitation_learning.' + tag + '.{}'.format(self.seed))
        self.restart_warmstart_every = 50
        self.work = str(current_dir) + '/work'
        self.save = os.path.join(self.work, tag)
        self.n_state, self.n_ctrl = self.env.true_dx.n_state, self.env.true_dx.n_ctrl
        self.n_sc = self.n_state + self.n_ctrl
        # experiment mpc
        self.true_q, self.true_p = self.env.true_dx.get_true_obj()
        self.env_params = self.env.true_dx.params
        self.T = self.env.mpc_T
        # remove if exists
        if os.path.exists(self.save):
            shutil.rmtree(self.save)
        os.makedirs(self.save)
        # NOTE
        self.is_lower_triangle = is_lower_triangle
        self.is_strange_observation = is_strange_observation
        if self.is_lower_triangle and not self.is_strange_observation:
            print("cost parameter is lower triangle")
            self.net = Pendulum_Net_cost_lower_triangle(self.n_sc, isrand=rand_init)
        elif not self.is_strange_observation:
            print("cost parameter is logit")
            self.net = Pendulum_Net_cost_logit(self.n_sc)
        elif self.is_strange_observation and not self.is_lower_triangle:
            print("cost parameter is logit and strange observation")
            self.net = Pendulum_Net_cost_logit_strange_obervation(self.n_sc, isrand=rand_init)
        else:
            print("cost parameter is lower triangle and strange observation")
            self.net = Pendulum_Net_cost_lower_triangle_strange_obervation(self.n_sc, isrand=rand_init)
        self.n_train = len(self.env.train_data)
        self.last_epoch = None
        self.n_data = None
        print("n train ", self.n_train)
        print("n batch ", self.n_batch)

    def make_data(self, data, shuffle=False):
        xs, us = data[:, :, :self.n_state], data[:, :, -self.n_ctrl:]
        x_inits = xs[:, 0]
        n_data = x_inits.shape[0]
        self.n_data = n_data
        ds = TupleDataset(x_inits, xs, us, xp.arange(0, n_data))
        loader = SerialIterator(ds, batch_size=self.n_batch, shuffle=shuffle)
        return ds, loader

    def dataset_loss(self, data_iter, warmstart=None):
        """ calculate loss and set warm start
        this loss is
        :param loader:
        :param warmstart:
        :return:
        """
        true_q, true_p = self.env.true_dx.get_true_obj()
        losses = []
        iter_before = data_iter.epoch
        while data_iter.epoch < iter_before + 1:
            next_batch = data_iter.next()
            next_batch = dataset.concat_examples(next_batch)
            x_inits = next_batch[0]
            xs = next_batch[1]
            us = next_batch[2]
            idxs = next_batch[3]
            n_batch = x_inits.shape[0]
            dx = self.env.true_dx
            if not self.is_lower_triangle and not self.is_strange_observation:
                print("not lower triangle and not strange observation")
                assert type(self.net) == Pendulum_Net_cost_logit
                q = F.sigmoid(self.net.learn_q_logit)
                p = F.sqrt(q) * self.net.learn_p
                _, pred_u = self.env.mpc(self.env.true_dx, x_inits, q, p,
                                         u_init=xp.transpose(warmstart[idxs], axes=(1, 0, 2)))
                pred_u = F.transpose(pred_u, axes=(1, 0, 2)).array
                warmstart[idxs] = pred_u
            elif not self.is_strange_observation:
                print("lower triangle and  not strange observation")
                assert type(self.net) == Pendulum_Net_cost_lower_triangle
                Q = xp.zeros((self.n_sc, self.n_sc))
                index_diag = xp.zeros((self.n_sc, self.n_sc), dtype=bool)
                xp.fill_diagonal(index_diag, True)
                index_not_diag = xp.zeros((self.n_sc, self.n_sc), dtype=bool)
                index_not_diag[xp.tril_indices(self.n_sc, -1)] = True
                Q = F.scatter_add(Q, xp.tril_indices(self.n_sc, -1), self.net.lower_without_diag)
                # index_not_diag = xp.zeros((self.n_state, self.n_state), dtype=bool)
                # index_not_diag[xp.tril_indices(self.n_state, -1)] = True
                # Q = F.scatter_add(Q, xp.tril_indices(self.n_state, -1), self.net.lower_without_diag)
                diag_q = F.sigmoid(self.net.learn_q_logit)
                Q = F.where(index_diag, diag_q, Q)
                Q = Q @ Q.T
                p = self.net.learn_p
                p = F.concat((p, xp.array(0.0).reshape(1, )), axis=0)
                _, pred_u = self.env.mpc_Q(self.env.true_dx, x_inits, Q, p,
                                           u_init=xp.transpose(warmstart[idxs], axes=(1, 0, 2)))
                pred_u = F.transpose(pred_u, axes=(1, 0, 2)).array
                warmstart[idxs] = pred_u

            elif not self.is_lower_triangle:
                print("not lower triangle and strange observation")
                assert type(self.net) == Pendulum_Net_cost_logit_strange_obervation
                q = F.sigmoid(self.net.learn_q_logit)
                p = F.sqrt(q) * self.net.learn_p
                # p = self.learn_p
                Q = xp.zeros((self.n_sc, self.n_sc))
                index_diag = xp.zeros((self.n_sc, self.n_sc), dtype=bool)
                xp.fill_diagonal(index_diag, True)
                Q = F.where(index_diag, q, Q)
                Q = OBSERVATION_MATRIX.T @ Q @ OBSERVATION_MATRIX
                p = p @ OBSERVATION_MATRIX
                _, pred_u = self.env.mpc_Q(self.env.true_dx, x_inits, Q, p,
                                           u_init=xp.transpose(warmstart[idxs], axes=(1, 0, 2)))
                pred_u = F.transpose(pred_u, axes=(1, 0, 2)).array
                warmstart[idxs] = pred_u
            else:
                print("lower triangle and strange observation")
                assert type(self.net) == Pendulum_Net_cost_lower_triangle_strange_obervation
                Q = xp.zeros((self.n_sc, self.n_sc))
                index_diag = xp.zeros((self.n_sc, self.n_sc), dtype=bool)
                xp.fill_diagonal(index_diag, True)
                index_not_diag = xp.zeros((self.n_sc, self.n_sc), dtype=bool)
                index_not_diag[xp.tril_indices(self.n_sc, -1)] = True
                Q = F.scatter_add(Q, xp.tril_indices(self.n_sc, -1), self.net.lower_without_diag)
                diag_q = F.sigmoid(self.net.learn_q_logit)
                Q = F.where(index_diag, diag_q, Q)
                Q = Q @ Q.T
                Q = OBSERVATION_MATRIX.T @ Q @ OBSERVATION_MATRIX
                p = self.net.learn_p
                p = p @ OBSERVATION_MATRIX
                _, pred_u = self.env.mpc_Q(self.env.true_dx, x_inits, Q, p,
                                           u_init=xp.transpose(warmstart[idxs], axes=(1, 0, 2)))
                pred_u = F.transpose(pred_u, axes=(1, 0, 2)).array
                warmstart[idxs] = pred_u

            assert pred_u.shape == us.shape
            squared_loss = (us - pred_u) * (us - pred_u)
            # print(squared_loss.shape)
            loss = xp.mean(squared_loss)
            losses.append(loss)
        loss = xp.stack(losses).mean()
        return loss

    def run(self):
        loss_names = ['epoch']
        loss_names.append('imitation_loss')
        if self.learn_dx:
            loss_names.append('sysid_loss')
        fname = os.path.join(self.save, 'train_losses.csv')
        train_loss_f = open(fname, 'w')
        train_loss_f.write('{}\n'.format(','.join(loss_names)))
        train_loss_f.flush()
        fname = os.path.join(self.save, 'val_test_losses.csv')
        vt_loss_f = open(fname, 'w')
        loss_names = ['epoch']
        loss_names += ['im_loss_val', 'im_loss_test']
        vt_loss_f.write('{}\n'.format(','.join(loss_names)))
        vt_loss_f.flush()
        if self.learn_cost:
            fname = os.path.join(self.save, 'cost_hist.csv')
            cost_f = open(fname, 'w')
            # first write answer
            cost_f.write(','.join(map(str, xp.concatenate((self.true_q, self.true_p)).tolist())))
            cost_f.write('\n')
            cost_f.flush()
        opt = chainer.optimizers.RMSprop(lr=1e-2, alpha=0.5)
        opt.setup(self.net)
        train_warm_start = xp.zeros((self.n_train, self.T, self.n_ctrl))
        val_warm_start = xp.zeros((self.env.val_data.shape[0], self.T, self.n_ctrl))
        test_warm_start = xp.zeros((self.env.test_data.shape[0], self.T, self.n_ctrl))

        train_data, train_iter = self.make_data(self.env.train_data[:self.n_train], shuffle=True)
        val_data, val_iter = self.make_data(self.env.val_data)
        test_data, test_iter = self.make_data(self.env.test_data)
        best_val_loss = None
        train_warmstart = xp.zeros((self.n_train, self.T, self.n_ctrl))
        val_warmstart = xp.zeros((self.env.val_data.shape[0], self.T, self.n_ctrl))
        test_warmstart = xp.zeros((self.env.test_data.shape[0], self.T, self.n_ctrl))
        true_q, true_p = self.env.true_dx.get_true_obj()
        cost_update_q = False
        learn_cost_round_robin_interval = 10

        while train_iter.epoch < self.n_epoch:
            print("triain_iter.epoch", train_iter.epoch_detail)
            if train_iter.epoch > 0 and train_iter.epoch % learn_cost_round_robin_interval == 0:
                cost_update_q = not cost_update_q
            self.net.cleargrads()
            if train_iter.epoch % self.restart_warmstart_every == 0:
                train_warm_start = xp.zeros_like(train_warm_start)
                val_warm_start = xp.zeros_like(val_warm_start)
                test_warm_start = xp.zeros_like(test_warm_start)
            next_batch = train_iter.next()
            next_batch = dataset.concat_examples(next_batch)
            xinits = next_batch[0]
            xs = next_batch[1]
            us = next_batch[2]
            idxs = next_batch[3]
            if self.learn_dx:
                assert False
            else:
                dx = self.env.true_dx
            nom_x, nom_u = self.net(xinits, self.env, train_warm_start[idxs])
            nom_u = F.transpose(nom_u, axes=(1, 0, 2))
            '''
            g = c.build_computational_graph([nom_u], remove_variable=True)
            with open('graph.dot', 'w') as o:
                o.write(g.dump())
            assert False
            '''
            train_warmstart[idxs] = nom_u.array
            assert type(us) != chainer.Variable
            assert nom_u.shape == us.shape
            squared_loss_u = (us - nom_u) * (us - nom_u)
            im_loss_u = F.mean(squared_loss_u)
            # print("im_loss shape", im_loss_u.shape)
            nom_x = F.transpose(nom_x, axes=(1, 0, 2))
            assert xs.shape == nom_x.shape, str(xs.shape) + " " + str(nom_x.shape)
            # squared_loss_x = (xs - nom_x) * (xs - nom_x)
            # im_loss_x = F.mean(squared_loss_x)
            # print("im_loss-x", im_loss_x.shape)
            # im_loss = im_loss_x + im_loss_u
            im_loss = im_loss_u
            t = [train_iter.epoch_detail, im_loss.array]
            t = ','.join(map(str, t))
            train_loss_f.write(t + '\n')
            train_loss_f.flush()
            print("train imtation loss", im_loss.array)
            im_loss.backward()
            if cost_update_q:
                print('only updating q')
                self.net.learn_p.update_rule.enabled = False
                self.net.learn_q_logit.update_rule.enabled = True
                if self.is_lower_triangle:
                    self.net.lower_without_diag.update_rule.enabled = True
            else:
                print('only updating p')
                self.net.learn_q_logit.update_rule.enabled = False
                self.net.learn_p.update_rule.enabled = True
                if self.is_lower_triangle:
                    self.net.lower_without_diag.update_rule.enabled = False
            '''
            if self.learn_cost:
                
                true_cat = F.concat((true_q, true_p), axis=0)
                _q = F.sigmoid(self.net.learn_q_logit).array
                _p = xp.sqrt(_q) * self.net.learn_p
                print("learn_q_logit", self.net.learn_q_logit)
                print("learn_p", self.net.learn_p)
                qp_cat = F.concat((_q, _p), axis=0)
                print(xp.array_str(F.stack((true_cat, qp_cat)).array, precision=5, suppress_small=True))
                cost_f.write(','.join(map(str, qp_cat.array)))
                cost_f.write('\n')
                cost_f.flush()
            '''
            opt.update()
            if train_iter.is_new_epoch:
                val_loss = self.dataset_loss(val_iter, val_warmstart)
                test_loss = self.dataset_loss(test_iter, test_warmstart)
                t = [train_iter.epoch, val_loss, test_loss]
                t = ','.join(map(str, t))
                vt_loss_f.write(t + '\n')
                vt_loss_f.flush()
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    fname = os.path.join(self.save, 'best.pkl')
                    print('Saving best model to {}'.format(fname))

                    with open(fname, 'wb') as f:
                        pkl.dump(self, f)


if __name__ == '__main__':
    fname = time.strftime("%Y%m%d-%H%M%S")
    print("Is parameter lower triangle matrix")
    lower_triangle = input("Yes/No: ")
    if lower_triangle == "Yes":
        is_lower_triangle = True
        fname += ":lower_triangle:"
    elif lower_triangle == "No":
        is_lower_triangle = False
        fname += ":not_lower_triangle:"
    else:
        assert False
    print("Is observation is strange ")
    strange_observation = input("Yes/No:")
    if strange_observation == "Yes":
        is_strange_observation = True
        fname += ":strange_observation:"
    elif strange_observation == "No":
        is_strange_observation = False
        fname += ":not_strange_observtion:"
    else:
        assert False
    if lower_triangle or strange_observation:
        print("Initialization is random or not?")
        rand_init = input("Yes/No: ")
        if rand_init == "Yes":
            fname += "random_initialization"
            random_init = True
        elif rand_init == "No":
            fname += "zero_initialization"
            random_init = False
        else:
            assert False
    n_epoch = int(input("Please enter n_epoch : "))
    n_batch = input("Please enter batch size [32]:") or "32"
    n_batch = int(n_batch)
    exp = IL_Exp(n_batch=n_batch, n_epoch=n_epoch, fname=fname, is_lower_triangle=is_lower_triangle,
                 is_strange_observation=is_strange_observation)
    exp.run()
