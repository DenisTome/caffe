# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:33:08 2016

@author: denitome
"""

import caffe
import numpy as np
import yaml

class MyCustomLayer(caffe.layers):
    
    def setup(self, bottom, top):
        # check input dimension
        if (len(bottom) != 1):
            raise Exception('This layer expects to receive as input only the heatmaps generated at the previous layer')
        if (len(top) != 1):
            raise Exception('This layer produces only one output blob')
        # TODO: check if it needs conversion
        self.num_joints = yaml.load(self.param_str)["njoints"]
        self.num_joints = self.njoints + 1
        print 'number of joints is %d' % self.num_joints
        # TODO: check that it works
        if (bottom[0].data.shape(1) != self.num_joints):
            raise Exception('Expected different number of heat-maps')
    
    def reshape(self, bottom, top):
        # Adjust the shapes of top blobs and internal buffers
        # to accommodate the shapes of the bottom blobs.
        
        # difference is shape of input
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # bottom has the same shape of input
        top[0].reshape(*bottom[0].data.shape)
    
    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data[...]
        