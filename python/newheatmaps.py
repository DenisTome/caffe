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
# TODO: remove
import cv2  
from scipy.stats import multivariate_normal
import scipy.io as sio
import sys
sys.path.insert(0,'python/manifold/')

import upright_cam as uc

class MyCustomLayer(caffe.Layer):
    
    def load_parameters(self):
        """Load trained parameters and camera rotation matrices"""
        Dpath = 'models/cpm_architecture/data/'
        par   = sio.loadmat(Dpath+'camera_param.mat')
        train = sio.loadmat(Dpath + 'train_basis_allactions.mat')
        R    = np.empty((4,3,3))
        R[0] = par['c1'][0,0]['R']
        R[1] = par['c2'][0,0]['R']
        R[2] = par['c3'][0,0]['R']
        R[3] = par['c4'][0,0]['R']
        return (R, train['e'], train['z'], train['a'].std(0)**-1 )
    
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
        # TODO: check if it's still needed
        self.train = bool(yaml.load(self.param_str)["train"])        
        (self.default_r, self.e, self.z, self.weights) = self.load_parameters()

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
    
    def findCoordinate(self, heatMap):
        """Given a heat-map of a squared dimension, identify the joint position as the 
        point with the highest likelihood. If there is not such a point, it returns
        the center of the heat-map as predicted joint position."""
        idx = np.where(heatMap == heatMap.max())
        x = idx[1][0]
        y = idx[0][0]
        if (heatMap[y,x]==0):
            x = int(heatMap.shape[1]/2)
            y = int(heatMap.shape[0]/2)
        return x,y
    
    def generateGaussian(self, pos, mean, Sigma):
        """Generate heat-map identifying the joint position. The probability distribution
        representing the joint position is a Gaussian with a given mean value representing
        the joint position and a given covariance matrix Sigma representing the uncertainty
        of the joint estimation.
        Pos: grid where the results are computed from
        Mean: gaussian mean value
        Sigma: gaussian convariance matrix"""
        rv = multivariate_normal([mean[1],mean[0]], Sigma)
        tmp = rv.pdf(pos)
        hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
        return hmap
    
    def generateHeatMaps(self, points):
        """Generate heat-maps for all the joint in the skeleton."""
        heatMaps = np.zeros((self.num_channels, self.input_size, self.input_size))
        sigma_sq = np.power(self.sigma,2)
        Sigma = [[sigma_sq,0],[0,sigma_sq]]
        
        x, y = np.mgrid[0:self.input_size, 0:self.input_size]
        pos = np.dstack((x, y))
        
        for i in range(self.num_joints):
            heatMaps[i,:,:] = self.generateGaussian(pos, points[i,:], Sigma)
        # generating last heat maps which contains all joint positions
        heatMaps[-1,:,:] =  heatMaps[0:heatMaps.shape[0]-1,:,:].max(axis=0)
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
        # on the mean value)==0 I would expect the mean values to be in (0,0) 
        heatmap_area = heatMap[area[1]:area[3],area[0]:area[2]]
        heatmap_area = np.divide(heatmap_area,np.sum(heatmap_area))
#        heatmap_area = np.divide(heatmap_area,np.nansum(heatmap_area))
        
        # extract covariance matrix
        flatten_hm = heatmap_area.flatten()
        x_coord = np.subtract(np.tile(range(area[0],area[2]), area[3]-area[1]),mean_value[0])
        y_coord = np.subtract(np.repeat(range(area[1],area[3]), area[2]-area[0]),mean_value[1])
        M = np.vstack((x_coord,y_coord))
    
        cov_matrix = np.dot(M,np.transpose(np.multiply(M,flatten_hm)))
        return mean_value, cov_matrix.flatten()
        
    
    def manifoldDataConversion(self, heatMaps, camera):
        # TODO: from the 2D points project the new 3D points
        points = np.zeros((self.num_joints,2))
        mean_values = np.zeros((self.num_joints,2))
        covariance_matrices = np.zeros((self.num_joints,4))
        # the last one is the overall heat-map
        for j in range(self.num_joints):
            mean_values[j,:],covariance_matrices[j,:] = self.findMeanCovariance(heatMaps[j,:,:])
        # Points contains the new projected points
        points = mean_values        
        return points
    
    def extractMetadata(self, channel):
        """Given a channel, extract the metadata about the frame."""
        # data written in c++ row-wise (from column 1 to column n)
        idx    = channel[0,0,0]
       c amera = channel[0,0,1]
        action = channel[0,0,2]
        person = channel[0,0,3]
        return idx, camera, action, person

    def centre(m):
        """Returns centered data in position (0,0)"""
        return ((m.T - m.mean(1)).T, m.mean(1))
          
    def normalise_data(w):
        """Normalise data by centering all the 2d skeletons (0,0) and by rescaling
        all of them such that the height is 2.0 (expressed in meters).
        """
        w = w.reshape(-1,2).transpose(1,0)
        d2,mean = centre(w)
        
        m2 = d2[1,:].min(0)/2.0
        m2 = m2-d2[1,:].max(0)/2.0
        crap = (m2 == 0)
        if crap:
            m2 = 1.0
        d2 /= m2
        return (d2, m2, mean)
    
    def project2D(r,z,a,e,R_c,scale):
        """Project 3D points into 2D."""
        mod = uc.build_model(a, e, z)
        p = uc.project_model(mod, R_c, r)
        return p
        
    def forward(self, bottom, top):
        """Forward data in the architecture to the following layer."""
        input_heatMaps = bottom[0].data[...]
        heatMaps = np.zeros((self.batch_size, self.num_channels, self.input_size, self.input_size))
        metadata = bottom[1].data[...]
        
        for b in range(self.batch_size):
            (_, camera, _, _) = self.extractMetadata(metadata[b,:,:,:])
            # get new points
            # TODO: retireve image number and perform the related actions
            points = self.manifoldDataConversion(input_heatMaps[b,:,:,:], camera)
            heatMaps[b,:,:,:] = self.generateHeatMaps(points)
            
            if (self.debug_mode):
                for j in range(self.num_channels):
                    name = '%s/tmp/batch_%d_beforeafter_%d.png' % (os.environ['HOME'], b, j)
                    if (np.max(input_heatMaps[b,j,:,:]) > 0):
                        rescaled_input = np.divide(input_heatMaps[b,j,:,:],np.max(input_heatMaps[b,j,:,:]))
                    else:
                        rescaled_input = input_heatMaps[b,j,:,:]
                    vis = np.concatenate((rescaled_input, heatMaps[b,j,:,:]), axis=1)
                    plt.imsave(name,vis)
                    if (np.max(input_heatMaps[b,j,:,:]) == 0):
                        name = '%s/tmp/exception_zero_%d_%d.png' % (os.environ['HOME'], b, j)
                        plt.imsave(name,vis)
        
        # TODO: change it to heatMaps
        top[0].data[...] = input_heatMaps
        #pass
    
    def backward(self, top, propagate_down, bottom):
        """Backward data in the learning phase. This layer does not propagate back information."""
        bottom[0].diff[...] = np.zeros(bottom[0].data.shape)
        #pass
        
        