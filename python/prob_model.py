# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:33:08 2016

@author: denitome
"""

import caffe
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal
from scipy import ndimage
import scipy.io as sio
import sys
import os
lib_path = os.environ['CAFFE_HOME_CPM']
sys.path.insert(0,'%s/python/manifold/' % lib_path)

import upright_cam as uc

class MyCustomLayer(caffe.Layer):

    def load_parameters(self):
        """Load trained parameters and camera rotation matrices"""
        Dpath = '%s/models/cpm_architecture/data/' % lib_path
        par   = sio.loadmat(Dpath+'camera_param.mat')
        # train = sio.loadmat(Dpath + 'train_basis_allactions.mat')
        R    = np.empty((4,3,3))
        R[0] = par['c1'][0,0]['R']
        R[1] = par['c2'][0,0]['R']
        R[2] = par['c3'][0,0]['R']
        R[3] = par['c4'][0,0]['R']

        prob_3d_pose_model = sio.loadmat(Dpath + 'prob_model_params.mat')
        mu = prob_3d_pose_model['mu']
        e = prob_3d_pose_model['e']
        sigma = prob_3d_pose_model['sigma']

        mu = mu.reshape(mu.shape[0], 3, 17)
        e = e.reshape(e.shape[0], e.shape[1], 3, 17)

        return R, e, mu, sigma

    def setup(self, bottom, top):
        """Set up the layer used for defining constraints to the expected inputs
        and for reading the parameters from the prototxt file.
        """
        # extract data from prototxt
        self.num_joints = yaml.load(self.param_str)["njoints"]
        self.num_channels = self.num_joints + 1
        self.batch_size = bottom[0].data.shape[0]
        self.input_size = bottom[0].data.shape[-1]
        self.sigma = yaml.load(self.param_str)["sigma"]
        self.debug_mode = yaml.load(self.param_str)["debug_mode"]
        self.max_area = yaml.load(self.param_str)["max_area"]
        self.percentage_max = yaml.load(self.param_str)["percentage_max"]*0.01
        self.train = bool(yaml.load(self.param_str)["train"])
        self.Lambda = yaml.load(self.param_str)["Lambda"]
        (self.default_r, self.e, self.mu, self.sigma_model) = self.load_parameters()

        # check input dimension
        if (len(bottom) != 2):
            raise Exception('This layer expects to receive as input the heatmaps generated at the previous layer plus metadata')
        if (len(top) != 1):
            raise Exception('This layer produces only one output blob')
        if (bottom[0].data.shape[1] != self.num_channels):
            raise Exception('Expected different number of heat-maps')
        if (bottom[1].data.shape[1] != 1):
            raise Exception('All metadata must be contained in a single channel')

    def reshape(self, bottom, top):
        """Reshape output according to the input. We want to keep the same dimensionality"""
        # Adjust the shapes of top blobs and internal buffers to accommodate the shapes of the bottom blobs.
        # Bottom has the same shape of input
        top[0].reshape(*bottom[0].data.shape)
        # self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

    def findCoordinate(self, heatMap):
        """Given a heat-map of a squared dimension, identify the joint position as the
        point with the highest likelihood. If there is not such a point, it returns
        the center of the heat-map as predicted joint position."""
        # heatMap = ndimage.gaussian_filter(heatMap, sigma=1)
        idx = np.where(heatMap == heatMap.max())
        try:
            x = idx[1][0]
            y = idx[0][0]
            if (heatMap[y, x] == 0):
                x = int(heatMap.shape[1] / 2)
                y = int(heatMap.shape[0] / 2)
        except:
            print 'FAILURE!\n\nGiven heat maps is a set of NaN\n\n'
            x = int(heatMap.shape[1] / 2)
            y = int(heatMap.shape[0] / 2)
        return x,y

    def generateGaussian(self, pos, mean, Sigma):
        """Generate heat-map identifying the joint position. The probability distribution
        representing the joint position is a Gaussian with a given mean value representing
        the joint position and a given covariance matrix Sigma representing the uncertainty
        of the joint estimation.
        Pos: grid where the results are computed from
        Mean: gaussian mean value
        Sigma: gaussian convariance matrix"""
        Sigma_matrix = np.copy(Sigma)
        Sigma_matrix[0,0] = Sigma[1,1]
        Sigma_matrix[1,1] = Sigma[0,0]
        rv = multivariate_normal([mean[1],mean[0]], Sigma_matrix)
        tmp = rv.pdf(pos)
        hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
        return hmap

    def generateHeatMaps(self, points, cov_matrices=False):
        """Generate heat-maps for all the joint in the skeleton."""
        heatMaps = np.zeros((self.num_channels, self.input_size, self.input_size))
        sigma_sq = np.power(self.sigma,2)
        Sigma = np.eye(2)*sigma_sq # np.array([[sigma_sq,0],[0,sigma_sq]])

        x, y = np.mgrid[0:self.input_size, 0:self.input_size]
        pos = np.dstack((x, y))
        for i in range(self.num_joints):
            heatMaps[i,:,:] = self.generateGaussian(pos, points[:,i], Sigma)
#            sio.savemat('/home/denitome/Desktop/res_h.mat',{'h':heatMaps})
#            raise Exception("cov error")
        # generating last heat maps which contains all joint positions
        heatMaps[-1,:,:] = np.maximum(1.0-heatMaps[0:heatMaps.shape[0]-1,:,:].max(axis=0), 0)
        return heatMaps

    def findMeanCovariance(self, heatMap):
        """Given a heat-map representing the likelihood of the joint in the image
        identify the joint position and the relative joint uncertainty represented
        as the covariance matrix."""
        mean_value = self.findCoordinate(heatMap)
        # at the beginning there might be heatmaps with all zero values
        if (heatMap[mean_value[1],mean_value[0]] == 0):
            sigma_sq = heatMap.shape[0]*heatMap.shape[0]
            M = np.array([[sigma_sq, 0], [0, sigma_sq]])
            return mean_value, M.flatten()

        idx = np.where(heatMap >= heatMap.max()* self.percentage_max)
        # there are some cases (especially at the beginning) where the heat-map is noise
        # and this condition may not be satisfied
        if (len(idx[0]) == 0):
            sigma_sq = heatMap.shape[0]*heatMap.shape[0]
            M = np.array([[sigma_sq, 0], [0, sigma_sq]])
            return mean_value, M.flatten()

        area = [np.min(idx[1]),np.min(idx[0]),np.max(idx[1]),np.max(idx[0])]
        if (np.max(np.abs([area[0]-mean_value[0],area[1]-mean_value[1],
                           area[2]-mean_value[0],area[3]-mean_value[1]])) > self.max_area):
            left = mean_value[0] - self.max_area
            if (left < 0):
                left = 0
            top = mean_value[1] - self.max_area
            if (top < 0):
                top = 0
            right = mean_value[0] + self.max_area
            if (right > heatMap.shape[0]):
                right = heatMap.shape[0]
            bottom = mean_value[1] + self.max_area
            if (bottom > heatMap.shape[1]):
                bottom = heatMap.shape[1]
            area = np.abs([left , top, right,bottom])

        # it does not matter that we are considering a small portion of the
        # heatmap (we are just estimating the covariance matrix which is independent
        # on the mean value, whatever it is).
        heatmap_area = heatMap[area[1]:area[3],area[0]:area[2]]

        # extract covariance matrix
        flatten_hm = heatmap_area.flatten()
        idx = np.where( flatten_hm < 0)
        flatten_hm[idx] = 0
        flatten_hm /= flatten_hm.sum()
        x_coord = np.subtract(np.tile(range(area[0],area[2]), area[3]-area[1]),mean_value[0])
        y_coord = np.subtract(np.repeat(range(area[1],area[3]), area[2]-area[0]),mean_value[1])
        M = np.vstack((x_coord,y_coord))

        cov_matrix = np.dot(M,np.transpose(np.multiply(M,flatten_hm)))
        return mean_value, cov_matrix.flatten()

    def centre(self, m):
        """Returns centered data in position (0,0)"""
        return ((m.T - m.mean(1)).T, m.mean(1))

    def normalise_data(self, w):
        """Normalise data by centering all the 2d skeletons (0,0) and by rescaling
        all of them such that the height is 2.0 (expressed in meters).
        """
        w = w.reshape(-1,2).transpose(1,0)
        (d2, mean) = self.centre(w)

        m2 = d2[1,:].min(0)/2.0
        m2 = m2-d2[1,:].max(0)/2.0
        crap = (m2 == 0)
        if crap:
            m2 = 1.0
        d2 /= m2
        return d2, m2, mean

    def project2D(self, mod, r, R_c, scale):
        """Project 3D points into 2D and return them in the same scale as the given
        2D predicted points."""
        p = uc.project_model(mod, R_c, r)
        p *= scale
        return p

    # ---------------------  NEW CODE  -------------------%

    def affine_estimate(self, w, mu, e, cam, sigma2, depth_reg=0.085, weights=np.zeros((0, 0, 0)), scale=10,
                        scale_mean=0.0016 * 1.8 * 1.2, scale_std=1.2 * 0, cap_scale=-0.00129):
        """Quick switch to allow reconstruction at unknown scale
        returns a,r and scale"""
        s = np.empty((sigma2.shape[0], sigma2.shape[1] + 1))
        s[:, 0] = scale_std
        s[:, 1:] = sigma2
        s[:, 1:-1] *= scale

        e2 = np.empty((e.shape[0], e.shape[1] + 1, 3, e.shape[3]))
        e2[:, 0] = mu
        e2[:, 1:] = e
        m = np.zeros_like(mu)

        res, a, r = uc.pick_e(w,
                              e2,
                              m,
                              cam, s, weights=weights,
                              interval=0.01, depth_reg=depth_reg, scale_prior=scale_mean)
        scale = a[:, :, 0]
        reestimate = scale > cap_scale
        m = mu * cap_scale
        print res.shape, a.shape, r.shape
        for i in xrange(scale.shape[0]):
            if reestimate[i].sum() > 0:
                ehat = e2[i:i + 1, 1:]
                mhat = m[i:i + 1]
                shat = s[i:i + 1, 1:]
                print i
                (res2, a2, r2) = uc.pick_e(w[reestimate[i]], ehat, mhat, cam, shat, weights=weights[reestimate[i]],
                                           interval=0.01, depth_reg=depth_reg,
                                           scale_prior=scale_mean)
                res[i:i + 1, reestimate[i]] = res2
                a[i:i + 1, reestimate[i], 1:] = a2
                a[i:i + 1, reestimate[i], 0] = cap_scale
                r[i:i + 1, :, reestimate[i]] = r2
        scale = a[:, :, 0]
        a = a[:, :, 1:] / a[:, :, 0][:, :, np.newaxis]
        return res, a, r, scale

    def build_model(self, a, e, s0):
        assert (s0.shape[1] == 3)
        assert (e.shape[2] == 3)
        assert (a.shape[1] == e.shape[1])
        out = np.einsum('...i,...ijk', a, e)
        out += s0
        return out

    def build_and_rot_model(self, a, e, s0, r):
        from numpy.core.umath_tests import matrix_multiply

        r2 = uc.upgrade_r(r.T).transpose(0, 2, 1)
        mod = self.build_model(a, e, s0)
        mod = matrix_multiply(r2, mod)
        return mod

    def better_rec(self, w, model, cam, s=1, weights=1, damp_z=1):
        """Quick switch to allow reconstruction at unknown scale
        returns a,r and scale"""
        from numpy.core.umath_tests import matrix_multiply
        proj = matrix_multiply(cam[np.newaxis], model)
        proj[:, :2] = (proj[:, :2] * s + w * weights) / (s + weights)
        proj[:, 2] *= damp_z
        out = matrix_multiply(cam.T[np.newaxis], proj)
        return out

    def create_rec(self, w2, mu, e, cam, sigma, weights, sigma_scaling=5.2, res_weight=1):
        from numpy.core.umath_tests import matrix_multiply
        res, a, r, scale = self.affine_estimate(w2, mu, e, cam, sigma, scale=sigma_scaling, weights=weights,
                                           depth_reg=0, cap_scale=-0.001, scale_mean=-0.003)
        Lambda = sigma
        remaining_dims = 3 * w2.shape[2] - e.shape[1]
        assert (remaining_dims >= 0)
        llambda = -np.log(Lambda)
        lgdet = np.sum(llambda[:, :-1], 1) + llambda[:, -1] * remaining_dims
        score = (res * res_weight + lgdet[:, np.newaxis] * (scale ** 2))
        best = np.argmin(score, 0)
        index = np.arange(best.shape[0])
        a2 = a[best, index]
        r2 = r[best, :, index].T
        rec = self.build_and_rot_model(a2, e[best], mu[best], r2)
        rec *= -np.abs(scale[best, index])[:, np.newaxis, np.newaxis]

        proj = matrix_multiply(cam[np.newaxis], rec)
        # proj[:, :2] = (w2 * 1.55 * weights) / (1.55 * weights)
        proj[:, :2] = (proj[:, :2] + w2 * 1.55 * weights) / (1 + 1.55 * weights)
        proj[:, 2] *= 1

        p2d = proj[:, :2]
        return p2d

    # ----------------------------------------------------------------- #

    def manifoldDataConversion(self, heatMaps, camera):
        """Apply the manifold model on the predicted 2D joint positions. This involves to
        preprocess the skeleton by normalising it in order to have something comparable across the
        people and the frameworks. This scale is then kept to go back to the original size.
        Given the new normalised 2D joint positions, the manifold is used to identify the 3D
        skeleton which is then projected back into 2D, defining the new 2D joint positions.
        Each joint is then considered individually to generate the relative heat map associated
        with it what will be used in the following layers in the architecture."""
#        points = np.zeros((2,self.num_joints))
        mean_values = np.zeros((self.num_joints, 2))
        covariance_matrices = np.zeros((self.num_joints, 4))

        # extract the joint positions with the relative unicertainty. Individually for each joint.
        for j in range(self.num_joints):
            mean_values[j,:], covariance_matrices[j,:] = self.findMeanCovariance(heatMaps[j])

        # normalise data; w are the normalised 2d points; s scaled;
        (w, s, mean) = self.normalise_data(mean_values.flatten())
        w = w[np.newaxis,:]

        # compute parameters
        weights = np.ones_like(w)
        points = self.create_rec(w, self.mu, self.e, self.default_r[camera], self.sigma_model, weights)[0]

        points *= s
        points += mean[:, np.newaxis]

        return points

    def extractMetadata(self, channel):
        """Given a channel, extract the metadata about the frame."""
        # data written in c++ row-wise (from column 1 to column n)
        # data have the Matlab format (index starting from 1)
        idx    = channel[0,0,0]
        camera = channel[0,0,1] - 1
        action = channel[0,0,2]
        person = channel[0,0,3]
        return (idx, camera, action, person)

    def forward(self, bottom, top):
        """Forward data in the architecture to the following layer."""
        input_heatMaps = bottom[0].data[...]
        heatMaps = np.zeros((self.batch_size, self.num_channels, self.input_size, self.input_size))
        metadata = bottom[1].data[...]

        # consider each image in the batch individually
        for b in range(self.batch_size):
            (_, camera, _, _) = self.extractMetadata(metadata[b])
            # get new points
            points  = self.manifoldDataConversion(input_heatMaps[b], camera)
            # TODO: remove this part
            heatMaps[b] = self.generateHeatMaps(points, cov_matrices=False)

            if (self.debug_mode):
                for j in range(self.num_channels):
                    name = '%s/tmp/batch_%d_beforeafter_%d.png' % (os.environ['HOME'], b, j)
                    if (np.max(input_heatMaps[b,j,:,:]) > 0):
                        rescaled_input = np.divide(input_heatMaps[b,j],np.max(input_heatMaps[b,j]))
                    else:
                        rescaled_input = input_heatMaps[b,j]
                    vis = np.concatenate((rescaled_input, heatMaps[b,j]), axis=1)
                    plt.imsave(name,vis)
                    if (np.max(input_heatMaps[b,j]) == 0):
                        name = '%s/tmp/exception_zero_%d_%d.png' % (os.environ['HOME'], b, j)
                        plt.imsave(name,vis)

        top[0].data[...] = heatMaps

    def backward(self, top, propagate_down, bottom):
        """Backward data in the learning phase. This layer does not propagate back information."""
        pass
        # bottom[0].diff[...] = np.zeros(bottom[0].data.shape)

