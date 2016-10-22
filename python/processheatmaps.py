# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:33:08 2016

@author: denitome
"""

import caffe
import numpy as np
import yaml

class MergeHeatMaps(caffe.Layer):
    
    def setup(self, bottom, top):
        """Set up the layer used for defining constraints to the expected inputs
        and for reading the parameters from the prototxt file.        
        """
        # extract data from prototxt
        self.init = yaml.load(self.param_str)["init"]               # either zero or avg
        self.lr = float(yaml.load(self.param_str)["learning_rate"])
        
        # get dimensionality
        self.batch_size = bottom[0].data.shape[0]
        self.num_channels = bottom[0].data.shape[1]
        self.input_size = bottom[0].data.shape[-1]
        
        # define weights
        self.blobs.add_blob(2)
        if (self.init == 'zero'):
            self.blobs[0].data[...] = [0, 1]
        else:
            self.blobs[0].data[...] = [0.5, 0.5]
        
        # check input dimension
        if (len(bottom) != 2):
            raise Exception('This layer expects to receive as input the heatmaps generated at the previous layer plus metadata')
        if (len(top) != 1):
            raise Exception('This layer produces only one output blob')
        if (bottom[0].data.shape != bottom[1].data.shape):
            raise Exception('Input data must have the same dimensionality')
    
    def reshape(self, bottom, top):
        """Reshape output according to the input. We want to keep the same dimensionality"""
        # Adjust the shapes of top blobs and internal buffers to accommodate the shapes of the bottom blobs.
        # Bottom has the same shape of input
        top[0].reshape(*bottom[0].data.shape)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
     
    def normaliseHeatMap(self, heatMaps_old, heatMaps_new, weights):
        heatMaps = (weights[0]*heatMaps_new + weights[1]*heatMaps_old)/weights.sum()
        max_val = heatMaps.max()
        max_before = heatMaps_old.max()
        return (max_before/max_val)*heatMaps
    
    def forward(self, bottom, top):
        """Forward data in the architecture to the following layer."""
        
#        if (self.blobs[0].data[...] < 0):
#            self.blobs[0].data[...] = 0
#        if (self.blobs[0].data[...] > 1):
#            self.blobs[0].data[...] = 1
            
        input_heatMaps    = bottom[0].data[...]
        input_heatMaps_mf = bottom[1].data[...]
        weights = self.blobs[0].data[...]
        heatMaps = np.zeros((self.batch_size, self.num_channels, self.input_size, self.input_size),
                            dtype = np.float32)
        
        # consider each image in the batch individually
        for b in range(self.batch_size):
            for c in range(self.num_channels):
                heatMaps[b,c] = self.normaliseHeatMap(input_heatMaps[b,c], 
                                                      input_heatMaps_mf[b,c], weights)
        top[0].data[...] = heatMaps
#        print 'merging weight: %r' % self.blobs[0].data[...]
    
    def backward(self, top, propagate_down, bottom):
        """Backward data in the learning phase. This layer does not propagate back information."""
        # TODO: should I consider the batch size
        raise Exception('Diff received is %r' % np.sum(top[0].diff[...]))
        avg = np.average(top[0].diff[...])
        self.blobs[0].diff[...] = self.lr*[avg, avg]       
        # bottom[0].diff[...] = np.zeros(bottom[0].data.shape)
        
        