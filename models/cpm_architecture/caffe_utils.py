# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:38:16 2016

@author: denitome
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import os
import caffe
from scipy.stats import multivariate_normal
import json

def load_configuration(offset=25, samplingRate=5, inputSize=368, outputSize=46,
                       njoints=17, sigma=7, sigma_center=21, stride=8, gpu=True):
    """Set-up default configurations"""
    NN = dict()
    NN['offset'] = offset
    NN['samplingRate'] = samplingRate
    NN['inputSize'] = inputSize
    NN['outputSize'] = outputSize
    NN['njoints'] = njoints
    NN['sigma'] = sigma
    NN['sigma_center'] = sigma_center
    NN['stride'] = stride
    NN['GPU'] = gpu
    return NN

def loadNet(folder_name, file_detail):
    """Load caffe model"""
    # defining models path
    caffe_home = os.environ['CAFFE_HOME_CPM']
    sub_dir = '%s/models/cpm_architecture/prototxt/caffemodel' % caffe_home
    # set file paths
    def_file = '%s/%s/pose_deploy.prototxt' % (sub_dir,folder_name)
    if isinstance(file_detail, basestring):
        model_file = '%s/%s/%s.caffemodel' % (sub_dir, folder_name, file_detail)
    else:
        model_file = '%s/%s/pose_iter_%d.caffemodel' % (sub_dir, folder_name, file_detail)
    # load caffe model
    net = caffe.Net(def_file, model_file, caffe.TEST)
    return net

def loadNetFromPath(caffemodel, prototxt):
    """Load caffe model"""
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return net

def setCaffeMode(gpu, device = 0):
    """Initialise caffe"""
    if gpu:
        caffe.set_mode_gpu()
        caffe.set_device(device)
    else:
        caffe.set_mode_cpu()
    
def getCenterJoint(joints):
    if (np.array(joints).shape[0] == 1):
        joints = xyJoints(np.array(joints).flatten())
    return joints[0]

def getBoundingBox(joints, center, offset, img_width, img_height):
    """Get bounding box containing all joints keeping as constant as possible
    the aspect ratio of the box"""
    max_x = -1
    max_y = -1
    for i in range(joints.shape[0]):
        j = joints[i]
        if (max_x < abs(j[0]-center[0])):
            max_x = abs(j[0]-center[0])
        if (max_y < abs(j[1]-center[1])):
            max_y = abs(j[1]-center[1])
    offset_x = max_x + offset
    offset_y = max_y + offset
    if (offset_x > offset_y):
        offset_y = offset_x
    else:
        offset_x = offset_y
    
    box_points = np.empty(4)
    # pos - 1 because joints are expressed in Matlab format
    box_points[:2] = center-[offset_x,offset_y]-1
    box_points[2:] = np.multiply([offset_x,offset_y],2)
    # check that are inside the image
    if (box_points[0] + box_points[2] > img_width):
        box_points[2] = img_width - box_points[0]
    if (box_points[0] < 0):
        box_points[0] = 0
    if (box_points[1] + box_points[3] > img_height):
        box_points[3] = img_height - box_points[1]
    if (box_points[1] < 0):
        box_points[1] = 0
        
    return np.round(box_points).astype(int)

def findPoint(heatMap):
    """Find maximum point in a given heat-map"""
    idx = np.where(heatMap == heatMap.max())
    x = idx[1][0]
    y = idx[0][0]
    return x,y

def generateGaussian(Sigma, input_size, position):
    """Generate gaussian in the specified position with the specified sigma value"""
    
    if not isinstance(Sigma,(np.ndarray, list)):
        # build covariance matrix
        sigma_sq = np.power(Sigma, 2)
        Sigma = np.eye(2)*sigma_sq      

    x, y = np.mgrid[0:input_size, 0:input_size]
    pos = np.dstack((x, y))
    rv = multivariate_normal([position[1], position[0]], Sigma)
    
    tmp = rv.pdf(pos)
    hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
    idx = np.where(hmap.flatten() <= np.exp(-4.6052))
    hmap.flatten()[idx] = 0
    return hmap

def cropImage(image, box_points):
    """Given a box with the format [x_min, y_min, width, height]
    it returnes the cropped image"""
    return image[box_points[1]:box_points[1]+box_points[3],box_points[0]:box_points[0]+box_points[2]]

def resizeImage(image, new_size):
    """Resize image to a defined size. The size if the same for width and height"""
    return cv2.resize(image, (new_size, new_size), interpolation = cv2.INTER_CUBIC)
#    return cv2.resize(image, (new_size, new_size), interpolation = cv2.INTER_LANCZOS4)

def getNumChannelsLayer(net, layer_name):
    return net.blobs[layer_name].channels

def netForward(net, imgch):
    """Run the model with the given input"""
    net.blobs['data'].data[...] = imgch
    net.forward() 

def getOutputLayer(net, layer_name):
    """Get output of layer_name layer"""
    return net.blobs.get(layer_name).data[0]
    
def restoreSize(channels, channels_size, box_points):
    """Given the channel, it resize and place the channel in the right position
    in order to have a final estimation in the original image coordinate system.
    Channel has the format (c x w x h) """
    num_channels = channels.shape[0]
    channels = channels.transpose(1,2,0)
#    new_img = imresize(channels, box_points[2],box_points[3])
    new_img = cv2.resize(channels, (box_points[2],box_points[3]), interpolation = cv2.INTER_LANCZOS4)
#    new_img = cv2.resize(channels, (box_points[2],box_points[3]), interpolation = cv2.INTER_CUBIC)

    reecreated_img = np.zeros((channels_size[0], channels_size[1], num_channels))
    reecreated_img[box_points[1]:box_points[1]+box_points[3],box_points[0]:box_points[0]+box_points[2]] = new_img
    return reecreated_img

def findPredictions(njoints, heatMaps):
    """Given the heat-maps returned from the NN, it returnes the joint positions.
    HeatMaps are in the format (w x h x c)"""
    assert(njoints == (heatMaps.shape[2]-1))
    predictions = np.empty((njoints, 2))
    for i in range(njoints):
        predictions[i] = findPoint(heatMaps[:,:,i])
    # +1 becuase we are considering the Matlab notation
    return (predictions + 1)

def getMasksFromFilename(file_name):
    """Given the name of a file from train or test set, it returns
    the masks about that specific frame"""
    while (file_name.find('/') >= 0):
        file_name = file_name[file_name.find('/')+1:]
    file_name = file_name[:file_name.find('.jpg')]
    data = file_name.split('_')
    
    fno = data[4]
    person = data[0]
    action = data[1]
    camera = data[3]
    return (fno, camera, action, person)

def generateMaskChannel(size, frame_num, camera, action, person):
    """Generate the metadata channel"""
    metadata = np.zeros((size, size))
    metadata[0,0] = frame_num
    metadata[0,1] = camera
    metadata[0,2] = action
    metadata[0,3] = person
    return metadata[:,:,np.newaxis]

def xyJoints(linearisedJoints):
    """Given a vector of joints it returns the joint positions in
    the [[x,y],[x,y]...] format"""
    num_elems = len(np.array(linearisedJoints).flatten())
    assert(num_elems >= (17*2))
    xy = linearisedJoints.reshape((num_elems/2, 2))
    return xy

def filterJoints(joints):
    """From the whole set of joints it removes those that are not used in 
    the error computation.
    Joints is in the format [[x,y],[x,y]...]"""
    joints_idx = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    new_joints = joints[joints_idx]
    return np.array(new_joints)

def plotHeatMap(heatMap, secondHeatMaps=[], title=False):
    """Plot single heat-map"""
    if (len(secondHeatMaps) == 0):
        plt.imshow(heatMap)
    else:
        plt.subplot(121), plt.imshow(heatMap)
        plt.subplot(122), plt.imshow(secondHeatMaps)
    if title:
        if (len(secondHeatMaps) == 0):
            plt.title(title)
        else:
            plt.suptitle(title)
    plt.axis('off')

def plotHeatMaps(heatMaps, secondHeatMaps=[], title=False):
    """Plot heat-maps one after the other"""
    # check format
    if ((heatMaps.shape[0]==heatMaps.shape[1]) or (heatMaps.shape[-1]==18)):
        heatMaps = heatMaps.transpose(2,0,1)
    if (len(secondHeatMaps) != 0):
        if ((secondHeatMaps.shape[0]==secondHeatMaps.shape[1]) or (secondHeatMaps.shape[-1]==18)):
            secondHeatMaps = secondHeatMaps.transpose(2,0,1)
    # plot heat-maps
    for i in range(heatMaps.shape[0]):
            if (len(secondHeatMaps) != 0):
                plotHeatMap(heatMaps[i], secondHeatMaps[i], title)
            else:
                plotHeatMap(heatMaps[i], secondHeatMaps, title)
            plt.waitforbuttonpress()

def loadJsonFile(json_file):
    """Load json file"""
    with open(json_file) as data_file:
        data_this = json.load(data_file)
        data = data_this['root']
        return (data, len(data))

def getCaffeCpm():
    """Get caffe com dir path"""
    return os.environ.get('CAFFE_HOME_CPM')+'/models/cpm_architecture'

def computeError(gt, pred):
    """Compute the euclidean distance between ground truth and predictions"""
    assert(pred.shape[0] > pred.shape[1])
    assert(gt.shape[0] == pred.shape[0])
    err = np.sqrt(np.power(gt-pred,2).sum(1)).mean()
    return err
    
