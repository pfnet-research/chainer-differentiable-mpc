#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Shun Arahata
"""
Make expert dataset
using il_env
"""
import os
import pathlib
import pickle as pkl

from il_env import IL_Env


def main(n_train, n_val, n_test):
    """

    :param n_train:
    :param n_val:
    :param n_test:
    :return:
    """
    current_dir = pathlib.Path(__file__).resolve().parent
    path = str(current_dir) + '/data/'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    env = IL_Env('pendulum', lqr_iter=500)
    print("call env populate data")
    env.populate_data(n_train=n_train, n_val=n_val, n_test=n_test, seed=0)
    print("finished calling env populate data")
    save = os.path.join(path, 'pendulum' + '.pkl')
    print('Saving data to {}'.format(save))
    with open(save, 'wb') as f:
        pkl.dump(env, f)


if __name__ == "__main__":
    n_train = int(input("Enter number of training set: "))
    n_val = int(input("Enter number of validation set: "))
    n_test = int(input("Enter number of validation set: "))
    main(n_train, n_val, n_test)
