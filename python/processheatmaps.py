# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:33:08 2016

@author: denitome
"""

import caffe
import os
import numpy as np
import yaml

class MergeHeatMaps(caffe.Layer):
    
    def setup(self, bottom, top):
        """Set up the layer used for defining constraints to the expected inputs
        and for reading the parameters from the prototxt file.        
        """
        # extract data from prototxt
        # self.debug_mode = yaml.load(self.param_str)["debug_mode"]
        self.init = yaml.load(self.param_str)["init"]               # either zero or avg
        
        # get dimensionality
        self.batch_size = bottom[0].data.shape[0]
        self.num_channels = bottom[0].data.shape[1]
        self.input_size = bottom[0].data.shape[-1]
        
#        # define weights
#        self.blobs.add_blob(self.num_channels)
#        if (self.init == 'zero'):
#            self.blobs[0].data[...] = np.zeros(self.num_channels)
#        else:
#            self.blobs[0].data[...] = np.array([0.5]*self.num_channels)
            
        # define weights
        self.blobs.add_blob(1)
        if (self.init == 'zero'):
            self.blobs[0].data[...] = 0
        else:
            self.blobs[0].data[...] = 0.5
        
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
     
    def normaliseHeatMap(self, heatMaps_old, heatMap):
        max_val = heatMap.max()
        max_before = heatMaps_old.max()
        return (max_before/max_val)*heatMap

    def forward(self, bottom, top):
        """Forward data in the architecture to the following layer."""
        
        # TODO:
        # contraint weights to be between 0 and 1 if I want to have just one param per channel
        # also check out to enforce that (truncating may not be a good solution)
        input_heatMaps    = bottom[0].data[...]
        input_heatMaps_mf = bottom[1].data[...]
        weights = self.blobs[0].data[...]
        heatMaps = np.zeros((self.batch_size, self.num_channels, self.input_size, self.input_size),
                            dtype = np.float32)
        
        # consider each image in the batch individually
        for b in range(self.batch_size):
            for c in range(self.num_channels):
                heatMaps[b,c] = self.normaliseHeatMap(input_heatMaps[b,c], 
                                                      weights*input_heatMaps_mf[b,c] +
                                                      (1-weights)*input_heatMaps[b,c])
        top[0].data[...] = heatMaps
    
    def backward(self, top, propagate_down, bottom):
        """Backward data in the learning phase. This layer does not propagate back information."""
        pass
        # bottom[0].diff[...] = np.zeros(bottom[0].data.shape)
        
        