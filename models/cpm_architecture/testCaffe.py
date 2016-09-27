# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""



import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import os
import caffe
from scipy.stats import multivariate_normal

def load_frame_data():
    curr_data = dict([('img_paths',[]), ('img_height',[]), ('img_width',[]), ('joint_self',[]), ('objpos',[])])
    # load data from the correct path
    if os.uname()[1] == 'mac':
        curr_data['img_paths'] = '/data/7_2_1_1_1.jpg'
    else:
        curr_data['img_paths'] = '/data/Human3.6M/Data/S7/Images/7_2_1_1_1.jpg'   
        
    curr_data['img_height'] = 1002.0
    curr_data['img_width'] = 1000.0 
    curr_data['joint_self'] = [[478.825, 426.586, 1.0], [502.959, 426.908, 1.0], [492.314, 513.686, 1.0],
                               [504.287, 598.628, 1.0], [487.845, 596.459, 1.0], [482.328, 592.055, 1.0],
                               [454.16, 426.272, 1.0],  [453.088, 515.142, 1.0], [458.693, 602.433, 1.0],
                               [443.096, 601.427, 1.0], [434.982, 596.402, 1.0], [478.826, 426.566, 1.0],
                               [489.997, 385.009, 1.0], [488.261, 332.892, 1.0], [483.348, 320.655, 1.0],
                               [488.648, 300.006, 1.0], [488.261, 332.892, 1.0], [466.81, 347.095, 1.0],
                               [452.303, 402.205, 1.0], [438.008, 447.47, 1.0],  [438.008, 447.47, 1.0],
                               [432.889, 444.12, 1.0],  [431.13, 466.512, 1.0],  [431.13, 466.512, 1.0],
                               [488.261, 332.892, 1.0], [511.148, 346.596, 1.0], [522.583, 400.663, 1.0], 
                               [521.042, 444.943, 1.0], [521.042, 444.943, 1.0], [512.944, 439.956, 1.0],
                               [521.327, 470.664, 1.0], [521.327, 470.664, 1.0]]
    curr_data['objpos'] = [478.825, 426.586]
    return curr_data

def load_configuration(gpu = True):
    NN = dict()
    NN['offset'] = 25
    NN['samplingRate'] = 5
    NN['offset'] = 25
    NN['inputSize'] = 368
    NN['outputSize'] = 46
    NN['njoints'] = 17
    NN['sigma'] = 7
    NN['sigma_center'] = 21
    NN['stride'] = 8
    NN['GPU'] = gpu
    return NN

def loadNet(dir_name, file_detail):
    # defining model
    caffe_home = os.environ['CAFFE_HOME_CPM']
    def_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/pose_deploy.prototxt' % (caffe_home, dir_name)
    if isinstance(file_detail, basestring):
        model_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/%s.caffemodel' % (caffe_home, dir_name, file_detail)
    else:
        model_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/pose_iter_%d.caffemodel' % (caffe_home, dir_name, file_detail)
    net = caffe.Net(def_file, model_file, caffe.TEST)
    return net
    

def filterJoints(joints_orig):
    joints_idx = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    joints = [0] * len(joints_idx)
    for j in range(len(joints_idx)):
        joints[j] = map(int, joints_orig[joints_idx[j]])
    return joints
    
def getBoundingBox(NN, data):
    joints = filterJoints(data['joint_self'])
    max_x = -1
    max_y = -1
    center = map(int, data['objpos'])
    for i in range(len(joints)):
        j = joints[i]
        if (max_x < abs(j[0]-center[0])):
            max_x = abs(j[0]-center[0])
        if (max_y < abs(j[1]-center[1])):
            max_y = abs(j[1]-center[1])
    offset_x = max_x + NN['offset']
    offset_y = max_y + NN['offset']
    if (offset_x > offset_y):
        offset_y = offset_x
    else:
        offset_x = offset_y
    if (center[0] + offset_x > data['img_width']):
        offset_x = data['img_width'] - center[0]
    if (center[0] - offset_x < 0):
        offset_x = center[0]
    if (center[1] + offset_y > data['img_height']):
        offset_y = data['img_height'] - center[1]
    if (center[1] - offset_y < 0):
        offset_y = center[1]
    return (offset_x, offset_y)
    
def findPoint(heatMap):
    idx = np.where(heatMap == heatMap.max())
    x = idx[1][0]
    y = idx[0][0]
    return x,y

def generateGaussian(NN, position, center_map = False):
    if (center_map):
        sigma_sq = np.power(NN['sigma_center'],2)
    else:
        sigma_sq = np.power(NN['sigma'],2)
        
    # build covariance matrix
    Sigma = [[sigma_sq,0],[0,sigma_sq]]
    x, y = np.mgrid[0:NN['inputSize'], 0:NN['inputSize']]
    pos = np.dstack((x, y))
    rv = multivariate_normal([position[1],position[0]], Sigma)
    
    tmp = rv.pdf(pos)
    hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
    return hmap

def mask_from_file(file_name):
    while (file_name.find('/') >= 0):
        file_name = file_name[file_name.find('/')+1:]
    file_name = file_name[:file_name.find('.jpg')]
    data = file_name.split('_')
    
    person = data[0]
    action = data[1]
    camera = data[3]
    fno = data[4]
    return (fno, camera, action, person)
    

def preprocess_image(NN, curr_data):
    # read the image
    img = cv2.imread(curr_data['img_paths'])
    
    center = map(int, curr_data['objpos'])
    bbox = getBoundingBox(NN, curr_data)
    
    joints = filterJoints(curr_data['joint_self'])

    # crop around person
    img_croppad = img[center[1]-bbox[1]:center[1]+bbox[1], center[0]-bbox[0]:center[0]+bbox[0]]
    
    # transform data
    offset_left = - (center[0] - bbox[0])
    offset_top = - (center[1] - bbox[1])
    center = np.sum([center, [offset_left, offset_top]], axis=0)
    for j in range(len(joints)):
        joints[j][0] += offset_left
        joints[j][1] += offset_top
        del joints[j][2]
        
    resizedImage = cv2.resize(img_croppad, (NN['inputSize'],NN['inputSize']), interpolation = cv2.INTER_CUBIC)
    fx = float(NN['inputSize'])/img_croppad.shape[1]
    fy = float(NN['inputSize'])/img_croppad.shape[0]
    center = map(int, np.multiply(center, [fx,fy]))
    for j in range(len(joints)):
        joints[j] = map(int, np.multiply(joints[j], [fx,fy]))
    
    return (resizedImage, joints, center)
    
def caffe_data(NN, net, img, center, data):
    NN['inputSize'] = net.blobs['data'].width
    num_channels = net.blobs['data'].channels

    # generate center heat-map
    center_hm = np.zeros((NN['inputSize'],NN['inputSize'],1))
    center_hm[:,:,0] = generateGaussian(NN, center, center_map=True)
    
    # generate channel for metadata
    (fno, camera, action, person) = mask_from_file(data['img_paths'])
    metadata_channel = np.zeros((NN['inputSize'],NN['inputSize'],1))
    metadata_channel[0,0,0] = fno
    metadata_channel[0,1,0] = camera
    metadata_channel[0,2,0] = action
    metadata_channel[0,3,0] = person
    
    input_img = np.divide(img, float(256))
    input_img = np.subtract(input_img, 0.5)
    
    if num_channels == 4:
        img4ch = np.concatenate((input_img, center_hm), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))
        # Give input
        net.blobs['data'].data[...] = img4ch    
    else:
        img5ch = np.concatenate((input_img, center_hm, metadata_channel), axis=2)
        img5ch = np.transpose(img5ch, (2, 0, 1))
        # Give input
        net.blobs['data'].data[...] = img5ch
    net.forward()

# Load data and configuration
NN = load_configuration(gpu = False)
data = load_frame_data()
#
## take caffe input data
(NN_img, joints, center) = preprocess_image(NN, data)
if NN['GPU']:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

# Load caffe model
net = loadNet('manifold_initialised', 'initialisation_zero')
net2 = loadNet('trial_5', 50000)
caffe_data(NN, net, NN_img, center, data)
caffe_data(NN, net2, NN_img, center, data)

# Get outputs
layer_names = ['conv7_stage1_new', 'Mconv5_stage2_new', 'Mconv5_stage3_new',
               'Mconv5_stage4_new', 'Mconv5_stage5_new', 'Mconv5_stage6_new']

labels = net.blobs.get(layer_names[0]).data
labels = np.reshape(labels,(18,NN['outputSize'],NN['outputSize']))
labels = np.transpose(labels, (1, 2, 0))
labels2 = net2.blobs.get(layer_names[-1]).data
labels2 = np.reshape(labels2,(18,NN['outputSize'],NN['outputSize']))
labels2 = np.transpose(labels2, (1, 2, 0))
    
plt.figure()
for h in range(NN['njoints'] + 1):
    plt.subplot(121), plt.title('old model')
    plt.imshow(labels2[:,:,h])
    plt.subplot(122), plt.title('new model')
    plt.imshow(labels[:,:,h])
    plt.draw()
    plt.waitforbuttonpress()

