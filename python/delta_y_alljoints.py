# -*- coding: utf-8 -*-
"""
Created on Apr 07 18:27 2017

@author: Denis Tome'
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import sys
from simulate_layer import manifoldDataConversion

nj = 17
nb = 25 # num bases
sigma = 1
lib_path = os.environ['CAFFE_HOME_CPM']
Dpath = '%s/models/cpm_architecture/data/' % lib_path

test_in_out_data = sio.loadmat('/home/denitome/Desktop/test_case_hm.mat')
Y = test_in_out_data['Y']
Y_hat, R = manifoldDataConversion(Y)
hm_in = test_in_out_data['Y_hm']
hm_out = test_in_out_data['Y_hat_hm']
test_prob_mod_data = sio.loadmat('/home/denitome/Desktop/prob_model_res.mat')
# e is going to be 3x25x3x17, we are just considering the case with 1 chart
# in create_rec there's the variable best saying which is the best one to use among the charts
# using best variable seelct right rotation r and bases e
e = np.array(test_prob_mod_data['e']).squeeze()
a = np.array(test_prob_mod_data['a']).squeeze()


def gaussian_heatmap(h, w, pos_x, pos_y, sigma_h=1, sigma_w=1, init=[]):
    """Compute the heat-map of size (w x h) with a gaussian distribution fit in
    position (pos_x, pos_y) and a convariance matix defined by the related sigma values.
    The resulting heat-map can be summed to a given heat-map init."""
    cov_matrix = np.eye(2) * ([sigma_h**2, sigma_w**2])

    x, y = np.mgrid[0:h, 0:w]
    pos = np.dstack((x, y))
    rv = multivariate_normal([pos_x, pos_y], cov_matrix)

    tmp = rv.pdf(pos)
    hmap = np.multiply(tmp, np.sqrt(np.power(2 * np.pi, 2) * np.linalg.det(cov_matrix)))
    idx = np.where(hmap.flatten() <= np.exp(-4.6052))
    hmap.flatten()[idx] = 0

    if np.size(init) == 0:
        return hmap

    assert (np.shape(init) == hmap.shape)
    hmap += init
    idx = np.where(hmap.flatten() > 1)
    hmap.flatten()[idx] = 1
    return hmap


def upgrade_r(r):
    """Upgrades complex parameterisation of planar rotation to tensor containing
    per frame 3x3 rotation matrices"""
    assert(r.ndim==2)
    #print r.shape
    assert(r.shape[1]==2) # Technically optional assert, but if this fails
                          # Data is probably transposed
    assert(np.all(np.isfinite(r)))
    norm=np.sqrt((r[:,:2]**2).sum(1))
    assert(np.all(norm>0))
    r/=norm[:,np.newaxis]
    assert(np.all(np.isfinite(r)))
    newr = np.zeros((r.shape[0], 3, 3))
    newr[:, :2, 0] = r[:, :2]
    #newr[:, :2, 0] *= norm[:,np.newaxis]
    newr[:, 2, 2] = 1
    newr[:, 1::-1, 1] = r[:, :2]
    #newr[:, 1::-1, 1] *= norm[:,np.newaxis]
    newr[:,0, 1] *= -1
    return newr


def get_d_y_hat_y(R):
    # build E*E_bar_trunc matrix
    r2 = upgrade_r(R.T).transpose((0, 2, 1)).squeeze()
    E = np.zeros((2*nj, nb))
    # tmp = np.empty((2*nj, 2*nj))

    for b in range(e.shape[0]):
        rot_e = np.dot(r2, e[b])
        E[:, b] = rot_e[:2].flatten()

    E_bar = np.zeros((2*nj + nb, nb))
    E_bar[:2*nj] = E
    E_bar[2*nj:] = np.eye(nb)*sigma
    E_bar_dagger = np.linalg.pinv(E_bar)
    E_bar_dagger_t = E_bar_dagger[:, :2*nj]

    grad_m = np.dot(E, E_bar_dagger_t)
    return grad_m



plt.figure(2)
plt.imshow(hm_in[0])

gt_pos = np.array([[ 21.96547088,  25.74329275,  26.43302096,  25.75795562,
         18.1876495 ,  20.65007279,  20.77926876,  21.98235491,
         23.15233835,  23.55859652,  23.56330801,  18.90838789,
         16.04871198,  18.23959672,  25.91823106,  27.67410767,
         29.43763563],
       [ 22.86180895,  22.94355865,  31.16839593,  41.02842797,
         22.17221567,  32.41282774,  42.105486  ,  16.30259985,
         10.07986668,   8.10301267,   5.92969572,  11.23519984,
         18.03582129,  21.69730372,  12.04103169,  18.99094312,
         25.39180452]])
Y[-1] = [28, 23]

h, w = 46, 46
lr = 0.6
num = 100
labels = np.zeros((nj, h, w))
for i in range(nj):
    labels[i] = gaussian_heatmap(h, w, gt_pos[1, i], gt_pos[0, i], sigma, sigma)
# plt.imshow(labels[0])
# plt.imshow(b_hat)

# generating "predicted" heat-map
for i in range(num):
    Y_hat, R = manifoldDataConversion(Y)
    d_y_hat_x = np.zeros(nj)
    d_y_hat_y = np.zeros(nj)
    err = 0
    for j in range(nj):
        b_hat = gaussian_heatmap(h, w, Y_hat[1, j], Y_hat[0, j], sigma, sigma)
        # plt.imshow(b_hat)

        # loss
        delta_b_hat = -(labels[j] - b_hat)
        # plt.imshow(delta_b_hat)
        d_b_hat_yp_x = (gaussian_heatmap(h, w, Y_hat[1, j], Y_hat[0, j] + 1, sigma, sigma) - \
                       gaussian_heatmap(h, w, Y_hat[1, j], Y_hat[0, j] - 1, sigma, sigma))/2
        # plt.imshow(d_b_hat_yp_x)
        d_b_hat_yp_y = (gaussian_heatmap(h, w, Y_hat[1, j] + 1, Y_hat[0, j], sigma, sigma) - \
                       gaussian_heatmap(h, w, Y_hat[1, j] - 1, Y_hat[0, j], sigma, sigma))/2
        # plt.imshow(d_b_hat_yp_y)
        d_y_hat_x[j] = np.dot(delta_b_hat.flatten(), d_b_hat_yp_x.flatten())
        d_y_hat_y[j] = np.dot(delta_b_hat.flatten(), d_b_hat_yp_y.flatten())

        Y_hat[0, j] -= lr * d_y_hat_x[j]
        Y_hat[1, j] -= lr * d_y_hat_y[j]

    d_y_hat = np.vstack((d_y_hat_x, d_y_hat_y)).T.flatten()
    T = get_d_y_hat_y(R)
    d_y = np.dot(T, d_y_hat)
    d_y_xy = np.reshape(d_y, (nj, 2))

    Y -= d_y_xy
    err = np.sqrt(np.sum(np.power(Y - gt_pos.T, 2)))
    print err

for j in range(nj):
    err = np.sqrt(np.sum(np.power(Y[j] - gt_pos[:, j], 2)))
    print 'joint %r err %r' % (j, err)

# TODO: use ONLY the first 17 heat-maps for using the gradient. compute all the new gradients according to that
#       and only at the end use all the gradients estimated for each of the heat-maps to estimate also the
#       gradient of the overall heat-map