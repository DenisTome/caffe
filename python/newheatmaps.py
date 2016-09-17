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

class MyCustomLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        # extract data from prototxt
        self.num_joints = yaml.load(self.param_str)["njoints"] + 1
        self.batch_size = bottom[0].data.shape[0]
        self.input_size = bottom[0].data.shape[-1]
        self.sigma = yaml.load(self.param_str)["sigma"]
        self.debug_mode = yaml.load(self.param_str)["debug_mode"]
        self.max_area = yaml.load(self.param_str)["max_area"]
        self.percentage_max = yaml.load(self.param_str)["percentage_max"]*0.01
        self.train = bool(yaml.load(self.param_str)["train"])

        # TODO: write the camera rotation matrices
        # is there any difference between training and testing at this point?
        # I am supposed to receive all the data I need here, without applying any
        # kind of mask.

        # TODO: load manifold data
        
        # check input dimension
        if (len(bottom) != 2):
            raise Exception('This layer expects to receive as input the heatmaps generated at the previous layer plus metadata')
        if (len(top) != 1):
            raise Exception('This layer produces only one output blob')
        if (bottom[0].data.shape[1] != self.num_joints):
            raise Exception('Expected different number of heat-maps')
        if (bottom[1].data.shape[1] != 1):
            raise Exception('All metadata must be contained in a single channel')
    
    def reshape(self, bottom, top):
        # Adjust the shapes of top blobs and internal buffers to accommodate the shapes of the bottom blobs.
        # Bottom has the same shape of input
        top[0].reshape(*bottom[0].data.shape)
    
    def findCoordinate(self, heatMap):
        idx = np.where(heatMap == heatMap.max())
        x = idx[1][0]
        y = idx[0][0]
        if (heatMap[y,x]==0):
            x = int(heatMap.shape[1]/2)
            y = int(heatMap.shape[0]/2)
        return x,y
    
    def generateGaussian(self, pos, mean, Sigma):
        rv = multivariate_normal([mean[1],mean[0]], Sigma)
        tmp = rv.pdf(pos)
        hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
        return hmap
    
    def generateHeatMaps(self, points):
        heatMaps = np.zeros((self.num_joints, self.input_size, self.input_size))
        sigma_sq = np.power(self.sigma,2)
        Sigma = [[sigma_sq,0],[0,sigma_sq]]
        
        x, y = np.mgrid[0:self.input_size, 0:self.input_size]
        pos = np.dstack((x, y))
        
        for i in range(self.num_joints - 1):
            heatMaps[i,:,:] = self.generateGaussian(pos, points[i,:], Sigma)
        # generating last heat maps which contains all joint positions
        heatMaps[-1,:,:] =  heatMaps[0:heatMaps.shape[0]-1,:,:].max(axis=0)
        return heatMaps
    
    def findMeanCovariance(self, heatMap):
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
        
    
    def manifoldDataConversion(self, heatMaps):
        # TODO: from the 2D points project the new 3D points
        points = np.zeros((self.num_joints,2))
        mean_values = np.zeros((self.num_joints,2))
        covariance_matrices = np.zeros((self.num_joints,4))
        # the last one is the overall heat-map
        for j in range(self.num_joints - 1):
            mean_values[j,:],covariance_matrices[j,:] = self.findMeanCovariance(heatMaps[j,:,:])
        # Points contains the new projected points
        points = mean_values        
        return points
    
    def extractMetadata(self, channel):
        # data written in c++ row-wise (from column 1 to column n)
        camera = -1
        action = -1
        person = -1      
        el1 = channel[0,0,0]
        el2 = channel[0,1,0]
        el3 = channel[0,2,0]
        raise Exception('%r\n%r\n%r' % (el1,el2,el3))
        return camera, action, person
        
    def forward(self, bottom, top):
        input_heatMaps = bottom[0].data[...]
        heatMaps = np.zeros((self.batch_size, self.num_joints, self.input_size, self.input_size))
        metadata = bottom[1].data[...]
        
        for b in range(self.batch_size):
            (camera,action,person) = self.extractMetadata(metadata[b,:,:,:])
            # get new points
            # TODO: retireve image number and perform the related actions
            points = self.manifoldDataConversion(input_heatMaps[b,:,:,:])
            heatMaps[b,:,:,:] = self.generateHeatMaps(points)
            
            if (self.debug_mode):
                for j in range(self.num_joints):
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
        bottom[0].diff[...] = np.zeros(bottom[0].data.shape)
        #pass
        
        