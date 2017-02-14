# -*- coding: utf-8 -*-
"""
Created on Feb 10 15:36 2017

@author: Denis Tome'
"""

import numpy as np
import matplotlib.pyplot as plt
import caffe
import caffe_utils as ut

ut.setCaffeMode(gpu=True)

MAX_ITERATIONS = 30
NUM_STAGES = 6
STEP_TRAIN = 1

solver = caffe.get_solver('models/cpm_architecture/prototxt/pose_solver.prototxt')
solver.net.copy_from('models/cpm_architecture/prototxt/caffemodel/manifold_diffarch3/pose_iter_22000.caffemodel')

# compute only the forward step
# solver.net.forward()

# compute forward and backward step by updating also the weights
losses = np.empty((MAX_ITERATIONS, NUM_STAGES + 1))
for i in range(0, MAX_ITERATIONS, STEP_TRAIN):
    solver.step(STEP_TRAIN)
    loss = 0.0
    for s in range(NUM_STAGES):
        losses[i, s] = float(solver.net.blobs[('loss_stage%r' % (s+1))].data)
        loss += losses[i, s]
    losses[i, -1] = loss
    print 'Loss iteration %r: %.3f' % (i, loss)
