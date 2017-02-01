# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""

import numpy as np
import scipy.io as sio
import sys

sys.path.insert(0, '/home/denitome/PythonProjects/')
import caffe_utils as ut
import argparse
import re
import cv2
import os

lib_path = os.environ['CAFFE_HOME_CPM']
sys.path.insert(0,'%s/python/manifold/' % lib_path)
import upright_cam as uc


def load_parameters():
    """Load trained parameters and camera rotation matrices"""
    lib_path = os.getenv('CAFFE_HOME_CPM')
    d_path = '%s/models/cpm_architecture/data/' % lib_path
    par = ut.sio.loadmat(d_path + 'camera_param.mat')
    train = ut.sio.loadmat(d_path + 'train_basis_allactions.mat')
    r_matrix = np.empty((4, 3, 3))
    r_matrix[0] = par['c1'][0, 0]['R']
    r_matrix[1] = par['c2'][0, 0]['R']
    r_matrix[2] = par['c3'][0, 0]['R']
    r_matrix[3] = par['c4'][0, 0]['R']
    return r_matrix, train['e'], train['z'], (train['a'].std(0)**-1)


def preprocess_images(nn_param, img, box_points):
    # manipulate image and joints for caffe
    img_croppad = ut.cropImage(img, box_points)
    resized_image = ut.resizeImage(img_croppad, nn_param['inputSize'])
    resized_image = np.divide(resized_image, float(256))
    resized_image = np.subtract(resized_image, 0.5)

    # generate labels and center channel
    input_size = nn_param['inputSize']
    center_channel = ut.generateGaussian(nn_param['sigma_center'], input_size, [input_size / 2, input_size / 2])

    metadata_ch = ut.generateMaskChannel(nn_param['inputSize'], 0, 1, 2, 1)
    imgch = np.concatenate((resized_image, center_channel[:, :, np.newaxis], metadata_ch), axis=2)

    imgch = np.transpose(imgch, (2, 0, 1))
    return imgch, img.shape[:2]


def postprocess_heatmaps(nn_param, out, image_size, bounding_box):
    heat_maps = ut.restoreSize(out, image_size, bounding_box)
    predictions = ut.findPredictions(nn_param['njoints'], heat_maps)
    return predictions


def normalise_data(w):
    """Normalise data by centering all the 2d skeletons (0,0) and by rescaling
    all of them such that the height is 2.0 (expressed in meters).
    """

    def centre(m):
        """Returns centered data in position (0,0)"""
        return (m.T - m.mean(1)).T, m.mean(1)

    w = w.reshape(-1, 2).transpose(1, 0)
    (d2, mean) = centre(w)

    m2 = d2[1, :].min(0) / 2.0
    m2 = m2 - d2[1, :].max(0) / 2.0
    crap = (m2 == 0)
    if crap:
        m2 = 1.0
    d2 /= m2

    return d2, m2, mean


def get_bounding_box(joints, image_path):
    image = cv2.imread(image_path)
    img_width = image.shape[1]
    img_height = image.shape[0]
    b_box = [joints[0].min(), joints[1].min(), joints[0].max(), joints[1].max()]

    offset_x = b_box[2] - b_box[0]
    offset_y = b_box[3] - b_box[1]
    if offset_x > offset_y:
        offset_y = offset_x
    else:
        offset_x = offset_y

    box_points = np.empty(4)
    box_points[0] = int(b_box[0] + (b_box[2] - b_box[0]) / 2 - offset_x / 2)
    box_points[1] = int(b_box[1] + (b_box[3] - b_box[1]) / 2 - offset_y / 2)
    box_points[2] = offset_x
    box_points[3] = offset_y
    # check that are inside the image
    if box_points[0] < 0:
        box_points[0] = 0
    if box_points[0] + box_points[2] > img_width:
        box_points[2] = img_width - box_points[0]
    if box_points[1] < 0:
        box_points[1] = 0
    if box_points[1] + box_points[3] > img_height:
        box_points[3] = img_height - box_points[1]
    return box_points.astype(int)

#
# def affine_estimate(w, mu, e, cam, sigma2, depth_reg=0.085, weights=np.zeros((0, 0, 0)), scale=10,
#                     scale_mean=0.0016 * 1.8 * 1.2, scale_std=1.2 * 0, cap_scale=-0.00129):
#     """Quick switch to allow reconstruction at unknown scale
#     returns a,r and scale"""
#     s = np.empty((sigma2.shape[0], sigma2.shape[1] + 1))
#     s[:, 0] = scale_std
#     s[:, 1:] = sigma2
#     s[:, 1:-1] *= scale
#
#     e2 = np.empty((e.shape[0], e.shape[1] + 1, 3, e.shape[3]))
#     e2[:, 0] = mu
#     e2[:, 1:] = e
#     m = np.zeros_like(mu)
#
#     res, a, r = uc.pick_e(w,
#                           e2,
#                           m,
#                           cam, s, weights=weights,
#                           interval=0.01, depth_reg=depth_reg, scale_prior=scale_mean)
#     scale = a[:, :, 0]
#     reestimate = scale > cap_scale
#     m = mu * cap_scale
#     print res.shape, a.shape, r.shape
#     for i in xrange(scale.shape[0]):
#         if reestimate[i].sum() > 0:
#             ehat = e2[i:i + 1, 1:]
#             mhat = m[i:i + 1]
#             shat = s[i:i + 1, 1:]
#             print i
#             (res2, a2, r2) = uc.pick_e(w[reestimate[i]], ehat, mhat, cam, shat, weights=weights[reestimate[i]],
#                                        interval=0.01, depth_reg=depth_reg,
#                                        scale_prior=scale_mean)
#             res[i:i + 1, reestimate[i]] = res2
#             a[i:i + 1, reestimate[i], 1:] = a2
#             a[i:i + 1, reestimate[i], 0] = cap_scale
#             r[i:i + 1, :, reestimate[i]] = r2
#     scale = a[:, :, 0]
#     a = a[:, :, 1:] / a[:, :, 0][:, :, np.newaxis]
#     return res, a, r, scale
#
#
# def build_model(a, e, s0):
#     assert (s0.shape[1] == 3)
#     assert (e.shape[2] == 3)
#     assert (a.shape[1] == e.shape[1])
#     out = np.einsum('...i,...ijk', a, e)
#     out += s0
#     return out
#
#
# def build_and_rot_model(a, e, s0, r):
#     from numpy.core.umath_tests import matrix_multiply
#
#     r2 = uc.upgrade_r(r.T).transpose(0, 2, 1)
#     mod = build_model(a, e, s0)
#     mod = matrix_multiply(r2, mod)
#     return mod
#
#
# def better_rec(w, model, cam, s=1, weights=1, damp_z=1):
#     """Quick switch to allow reconstruction at unknown scale
#     returns a,r and scale"""
#     from numpy.core.umath_tests import matrix_multiply
#     proj = matrix_multiply(cam[np.newaxis], model)
#     proj[:, :2] = (proj[:, :2] * s + w * weights) / (s + weights)
#     proj[:, 2] *= damp_z
#     out = matrix_multiply(cam.T[np.newaxis], proj)
#     return out
#
#
# def renorm_gt(gt):
#     """Compel gt data to have mean joint length of one"""
#     p = np.asarray([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
#                     [0, 7], [7, 8], [8, 9], [9, 10],
#                     [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]).T
#     scale = np.sqrt(((gt[:, :, p[0]] - gt[:, :, p[1]])**2).sum(2).sum(1))
#     return gt/scale[:, np.newaxis, np.newaxis], scale
#
#
# def create_rec(w2, mu, e, cam, sigma, weights, sigma_scaling=5.2, res_weight=1):
#     res, a, r, scale = affine_estimate(w2, mu, e, cam, sigma, scale=sigma_scaling, weights=weights,
#                                        depth_reg=0, cap_scale=-0.001, scale_mean=-0.003)
#     lambda_val = sigma
#     remaining_dims = 3 * w2.shape[2] - e.shape[1]
#     assert (remaining_dims >= 0)
#     llambda = -np.log(lambda_val)
#     lgdet = np.sum(llambda[:, :-1], 1) + llambda[:, -1] * remaining_dims
#     score = (res * res_weight + lgdet[:, np.newaxis] * (scale ** 2))
#     best = np.argmin(score, 0)
#     index = np.arange(best.shape[0])
#     a2 = a[best, index]
#     r2 = r[best, :, index].T
#     rec = build_and_rot_model(a2, e[best], mu[best], r2)
#     rec *= -np.abs(scale[best, index])[:, np.newaxis, np.newaxis]
#     rec = better_rec(w2, rec, cam, 1, 1.55 * weights, 1) * -1
#     rec, _ = renorm_gt(rec)
#     rec *= 0.97
#     return tmp


def test_images(net, nn_config, joints, input_dir, output_dir):
    files = [f for f in ut.os.listdir(input_dir) if f.endswith('.jpg')]
    files.sort()

    idx = 0
    for image_name in files:
        print 'Image %d\n' % idx
        test_image = input_dir + image_name
        image = cv2.imread(test_image)

        bbox = get_bounding_box(joints[:,:,idx], test_image)
        in_ch, img_size = preprocess_images(nn_config, image, bbox)
        ut.netForward(net, in_ch)
        layer_name = 'Mconv5_stage6_new'
        out_ch = ut.getOutputLayer(net, layer_name)
        preds = postprocess_heatmaps(nn_config, out_ch, img_size, bbox)
        res_img = ut.plotImageJoints(image, preds)
        #ut.plt.waitforbuttonpress()
        ut.plt.close()
        ut.plt.imsave(output_dir + image_name, res_img)
        #print 'Predictions:%r' % preds

        # extract 3D information
        lambda_val = 0.05
        (default_r, e, z, weights) = load_parameters()
        (w, s, mean) = normalise_data(preds.flatten())
        w = w[np.newaxis, :]
        camera = 1
        (a, r) = uc.estimate_a_and_r(w, e, z, default_r[camera], lambda_val * weights)
        for j in xrange(10):
            r = uc.reestimate_r(w, z, a, e, default_r[camera], r)
            (a, res) = uc.reestimate_a(w, e, z, r, default_r[camera], lambda_val * weights)
        mod = uc.build_model(a, e, z).squeeze()
        save_name = output_dir+image_name+'_3d.jpg'
        #ut.plot3DJoints_full(-mod, pbaspect=True, save_img=save_name)
        ut.plot3DJoints(-mod, save_img=save_name)
        #ut.plt.waitforbuttonpress()
        ut.plt.close()
        idx += 1


def main():
    # IMPORTANT: the 3D information that is extracted here is extracted using the OLD MODEL
    #            using a SINGLE CHART. This is done because the manifold layer is still inside the architecture
    #            and if we import the new modules (upright_cam, etc.) it will fail in estimating the 2D positions.
    #            Fix: run the 2D estimation first and then separatly the 3D estimation importing the latest modules.
    #            If needed look at the code in $CAFFE_HOME_CPM/python/prob_model.py to see what to import.
    leeds_dir = '/data/LEEDS/original'
    caffemodel = os.getenv('CAFFE_HOME_CPM') + \
                 '/models/cpm_architecture/prototxt/caffemodel/manifold_diffarch3/to_test/pose_iter_22000.caffemodel'
    prototxt = os.getenv('CAFFE_HOME_CPM') + '/models/cpm_architecture/prototxt/pose_deploy_singleimg.prototxt'

    nn = ut.load_configuration(gpu=True)
    ut.setCaffeMode(nn['GPU'])
    net = ut.loadNetFromPath(caffemodel, prototxt)

    images_dir = leeds_dir + '/images/'
    # images_dir = leeds_dir + '/tmp/'
    res_dir = leeds_dir + '/processed/'
    # res_dir = leeds_dir + '/processed_tmp/'
    gt_2d = sio.loadmat(leeds_dir + '/joints.mat')['joints']
    test_images(net, nn, gt_2d, images_dir, res_dir)


if __name__ == '__main__':
    main()