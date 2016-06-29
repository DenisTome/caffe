# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:33:08 2016

@author: denitome
"""

import caffe
import numpy as np
import yaml
#import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class MyCustomLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        # extract data from prototxt
        self.num_joints = yaml.load(self.param_str)["njoints"] + 1
        self.batch_size = bottom[0].data.shape[0]
        self.input_size = bottom[0].data.shape[-1]
        self.sigma = yaml.load(self.param_str)["sigma"]
        
        # check input dimension
        if (len(bottom) != 1):
            raise Exception('This layer expects to receive as input only the heatmaps generated at the previous layer')
        if (len(top) != 1):
            raise Exception('This layer produces only one output blob')
        if (bottom[0].data.shape[1] != self.num_joints):
            raise Exception('Expected different number of heat-maps')
    
    def reshape(self, bottom, top):
        # Adjust the shapes of top blobs and internal buffers to accommodate the shapes of the bottom blobs.
        # BGottom has the same shape of input
        top[0].reshape(*bottom[0].data.shape)
    
    def findCoordinate(self, heatMap):
        idx = np.where(heatMap == heatMap.max())
        x = idx[1][0]
        y = idx[0][0]
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
        
        for i in range(self.num_joints):
            heatMaps[i,:,:] = self.generateGaussian(pos, points[i,:], Sigma)
            
        # TODO: compute the last heat maps considering the maximum to be 1 (as in caffe code)
        return heatMaps
    
    def manifoldDataConversion(self, points):
        # TODO: from the 2D points project the new 3D points
        return points
    
    def forward(self, bottom, top):
        heatMaps = bottom[0].data[...]
        
        for b in range(self.batch_size):
            points = np.zeros((self.num_joints,2))
            # the last one is the overall heat-map
            for j in range(self.num_joints - 1):
                points[j,:] = self.findCoordinate(heatMaps[b,j,:,:])
            
            # get new points
            points = self.manifoldDataConversion(points)
            
            # TODO: generate the overall heatmap
            heatMaps[b,:,:,:] = self.generateHeatMaps(points)
            #new = self.generateHeatMaps(points)
#            for j in range(self.num_joints - 1):
#                name1 = '/home/denitome/before_%d.png' % j
#                name2 = '/home/denitome/after_%d.png' % j
#                plt.imsave(name1,heatMaps[b,j,:,:])
#                plt.imsave(name2,new[j,:,:])
                
        top[0].data[...] = heatMaps
        
        