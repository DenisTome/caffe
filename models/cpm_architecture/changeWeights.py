# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:50:40 2016

@author: denis
"""

import os
import caffe
import numpy as np

def loadNet(dir_name, iter_num, def_file_name = False):
    # defining model
    caffe_home = os.environ['CAFFE_HOME_CPM']
    if not def_file_name:
        def_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/pose_deploy.prototxt' % (caffe_home, dir_name)
    else:
        def_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/%s.prototxt' % (caffe_home, dir_name, def_file_name)
       
    model_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/pose_iter_%d.caffemodel' % (caffe_home, dir_name, iter_num)
    net = caffe.Net(def_file, model_file, caffe.TEST)
    return net

def saveNet(net, dir_name, file_name):
    caffe_home = os.environ['CAFFE_HOME_CPM']
    final_path = '%s/models/cpm_architecture/prototxt/caffemodel/%s' % (caffe_home, dir_name)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    net.save('%s/%s.caffemodel' % (final_path, file_name))

def getNewLayers(nstages = 6):
    layers_old = [""]* (nstages-1)
    layers_new = [""]* (nstages-1)
    for s in range(0,nstages-1):
        layers_old[s] = 'Mconv1_stage%d_new' % (s+2)
        layers_new[s] = 'Mconv1_stage%d_new_mf' % (s+2)
    return (layers_old, layers_new)

def isLayerToAnalyse(layer_name, changed_layers):
    if ('image' in layer_name):
        return False
    if ('pool' in layer_name):
        return False
    if ('relu' in layer_name):
        return False
    if ('concat' in layer_name):
        return False
    if (layer_name in changed_layers):
        return False
    return True
    
caffe.set_mode_cpu()
(layer_names_old, layer_names_new) = getNewLayers()

# LOAD MODELS
net1 = loadNet('trial_5', 50000)
net2 = loadNet('trial_5', 50000, 'pose_deploy_manifold')

# DEFINE NEW WEIGHTS
heatmaps_ch_size = 18
for layer in zip(layer_names_old,layer_names_new):
    # read weights
    weights = net1.params[layer[0]][0].data
    # define new weights
#    init_weights = weights[:,32:32+heatmaps_ch_size]/2
    init_weights = weights[:,32:32+heatmaps_ch_size]
#    weights[:,32:32+heatmaps_ch_size] = init_weights
#    new_weights = np.concatenate((weights,init_weights),axis=1)
    new_weights = np.concatenate((weights,np.zeros(init_weights.shape)),axis=1)
    net2.params[layer[1]][0].data[...] = new_weights
    net2.params[layer[1]][1].data[...] = net1.params[layer[0]][1].data

#CHECK THAT THE OTHER LAYERS HAVE THE CORRECT WEIGHTS
print '\n\nAnalazing model weights: comparison'
all_layer_names = list(net1._layer_names)
for layer_name in all_layer_names:
    if not isLayerToAnalyse(layer_name, layer_names_old):
        continue
    weights_net1 = net1.params[layer_name][0].data
    weights_net2 = net2.params[layer_name][0].data
    biases_net1 = net1.params[layer_name][1].data
    biases_net2 = net2.params[layer_name][1].data
    same_w = np.array_equal(weights_net1, weights_net2)
    same_b = np.array_equal(biases_net1, biases_net2)
    if not(same_w and same_b):
        print 'Layer %s has unmatching weights; weights %r, bias %r' % (layer_name, same_w, same_b)
print 'Analysis complete'

saveNet(net2, 'manifold_initialised', 'initialisation_zero')


