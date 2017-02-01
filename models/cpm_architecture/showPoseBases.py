# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""

import numpy as np
import scipy.io as sio
import sys
import os

home_dir = os.environ['CAFFE_HOME_CPM']
sys.path.insert(0, '%s/models/cpm_architecture/' % home_dir)
import caffe_utils as ut


def load_data():
    Dpath = '%s/models/cpm_architecture/data/' % os.environ['CAFFE_HOME_CPM']
    prob_3d_pose_model = sio.loadmat(Dpath + 'prob_model_params.mat')

    mu = prob_3d_pose_model['mu']
    e = prob_3d_pose_model['e']
    mu = mu.reshape(mu.shape[0], 3, 17)
    e = e.reshape(e.shape[0], e.shape[1], 3, 17)
    return mu,e

def show_base(chart, base_n):
    mu, e = load_data()
    #ut.plot3DJoints_full(mu[chart] + e[chart, base_n])
    #ut.plot3DJoints_full(mu[chart])
    ut.plot3DJoints_full(e[chart, base_n])
    ut.plt.show()

show_base(0,10)