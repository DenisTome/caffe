# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:25:18 2016

@author: denitome
"""

import os
import re
import caffe
import cv2
import json
import numpy as np
from scipy.stats import multivariate_normal
import scipy.io as sio
import matplotlib.pyplot as plt

# general settings
samplingRate = 150
offset = 25
inputSizeNN = 368
outputSizeNN = 46
joints_idx = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
sigma = 7
sigma_center = 21
stride = 8
verbose = False
fn_notification = 25
iter_start_from = 45000
device_id = 0

def filterJoints(joints_orig):
    joints = [0] * len(joints_idx)
    for j in range(len(joints_idx)):
        joints[j] = map(int, joints_orig[joints_idx[j]])
    return joints
    
def getBoundingBox(data):
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
    offset_x = max_x + offset
    offset_y = max_y + offset
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

def visualiseImage(image, bbox, center, joints):
    img = image.copy()
    img_croppad = image.copy()
    img_croppad = img_croppad[center[1]-bbox[1]:center[1]+bbox[1], center[0]-bbox[0]:center[0]+bbox[0]]
    
    cv2.line(img, (center[0]-bbox[0],center[1]-bbox[1]), (center[0]+bbox[0],center[1]-bbox[1]), (255, 0, 0), 2)
    cv2.line(img, (center[0]+bbox[0],center[1]-bbox[1]), (center[0]+bbox[0],center[1]+bbox[1]), (255, 0, 0), 2)
    cv2.line(img, (center[0]+bbox[0],center[1]+bbox[1]), (center[0]-bbox[0],center[1]+bbox[1]), (255, 0, 0), 2)
    cv2.line(img, (center[0]-bbox[0],center[1]+bbox[1]), (center[0]-bbox[0],center[1]-bbox[1]), (255, 0, 0), 2)
   
    for j in range(len(joints)):
        cv2.circle(img_croppad, (int(joints[j][0]), int(joints[j][1])), 3, (0, 255, 255), -1)
    
    cv2.imshow('Selected image',img)
    cv2.imshow('Cropped image',img_croppad)
    cv2.waitKey()
       
def generateGaussian(pos, mean, Sigma):
    rv = multivariate_normal([mean[1],mean[0]], Sigma)
    tmp = rv.pdf(pos)
    hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
    return hmap

def generateHeatMaps(center, joints):
    num_joints = len(joints_idx)
    heatMaps = np.zeros((inputSizeNN,inputSizeNN,num_joints+1))
    sigma_sq = np.power(sigma,2)
    Sigma = [[sigma_sq,0],[0,sigma_sq]]
    
    x, y = np.mgrid[0:368, 0:368]
    pos = np.dstack((x, y))
    
    # heatmaps representing the position of the joints
    for i in range(num_joints):
        heatMaps[:,:,i] = generateGaussian(pos, joints[i], Sigma)
    # generating last heat maps which contains all joint positions
    joints_heatmaps = heatMaps[:,:,0:heatMaps.shape[2]-1]
    heatMaps[:,:,-1] = joints_heatmaps.max(axis=2)
    
    # heatmap to be added to the RGB image
    sigma_sq = np.power(sigma_center,2)
    Sigma = [[sigma_sq,0],[0,sigma_sq]]
    center_hm = np.zeros((inputSizeNN,inputSizeNN,1))
    center_hm[:,:,0] = generateGaussian(pos, center, Sigma)
    return heatMaps, center_hm

def findPoint(heatMap):
    idx = np.where(heatMap == heatMap.max())
    x = idx[1][0]
    y = idx[0][0]
    return x,y

def generateChannel(frame_number, masks):
    # extract data; all the index problem about the masks are hendled inside the manifold layer
    # we just need the correct frame index here
    camera = masks['mask_camera'][0,frame_number-1]
    action = masks['mask_action'][0,frame_number-1]
    person = masks['mask_person'][0,frame_number-1]
    
    metadata = np.zeros((inputSizeNN,inputSizeNN))
    metadata[0,0,0] = frame_number
    metadata[0,1,0] = camera
    metadata[0,2,0] = action
    metadata[0,3,0] = person
    
    metadata = metadata[:,:,np.newaxis]
    return metadata
    

def runCaffeOnModel(data, model_dir, def_file, idx, masks):
    layer_names = ['conv7_stage1_new', 'Mconv5_stage2_new', 'Mconv5_stage3_new',
               'Mconv5_stage4_new', 'Mconv5_stage5_new', 'Mconv5_stage6_new']
    loss_stage = np.zeros(len(layer_names))
    mpepj_model = []
    
    iterNumber = getIter(model_dir)
    print '-------------------------------'
    print '  Evaluating iteration: %d' % iterNumber
    print '-------------------------------'
    
    print 'Loading model...'
    net = caffe.Net(def_file, model_dir, caffe.TEST)
    print 'Done.'
    for i in range(len(idx)):
        if (np.mod(i,fn_notification) == 0):
            print 'Iteration %d out of %d' % (i+1, len(idx))
        fno = idx[i]
        if (not data[fno]['isValidation']):
			continue

        curr_data = data[fno]
        center = map(int, curr_data['objpos'])
        bbox = getBoundingBox(curr_data)
              
        # take data
        img = cv2.imread(curr_data['img_paths'])
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
        
        # visualize data
#        if (verbose):
#            visualiseImage(img, bbox, map(int, curr_data['objpos']), joints)
        
        # resize image and update joint positions
        resizedImage = cv2.resize(img_croppad, (inputSizeNN,inputSizeNN), interpolation = cv2.INTER_CUBIC)
        fx = float(inputSizeNN)/img_croppad.shape[1]
        fy = float(inputSizeNN)/img_croppad.shape[0]
        assert(fx != 0)
        assert(fy != 0)
        
        center = map(int, np.multiply(center, [fx,fy]))
        for j in range(len(joints)):
            joints[j] = map(int, np.multiply(joints[j], [fx,fy]))
        if (verbose):
            tmp = resizedImage.copy()
            for j in range(len(joints)):
                cv2.circle(tmp, (joints[j][0], joints[j][1]), 3, (0, 255, 255), -1)
            cv2.circle(tmp, (center[0], center[1]), 3, (255, 255, 255), -1)
            plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
            plt.waitforbuttonpress()
        
        labels, center = generateHeatMaps(center, joints)
        # TODO: check that annolist_index is loaded correctly
        metadata = generateChannel(data['annolist_index'], masks)
        
        if (verbose):
            for j in range(len(joints) + 1):
                plt.imshow(labels[:,:,j])
                plt.waitforbuttonpress()
                
        resizedImage = np.divide(resizedImage,float(256))
        resizedImage = np.subtract(resizedImage, 0.5)

        # TODO: check that everything works fine here
        img5ch = np.concatenate((resizedImage, center, metadata), axis=2)
        img5ch = np.transpose(img5ch, (2, 0, 1))
        
        net.blobs['data'].data[...] = img5ch
        net.forward()
    
        for l in range(len(layer_names)):
            heatMaps = net.blobs.get(layer_names[l]).data
            num_channels = heatMaps.shape[1]
            heatMaps = np.reshape(heatMaps,(num_channels,outputSizeNN,outputSizeNN))
            heatMaps = np.transpose(heatMaps, (1, 2, 0))
            
            # reshape the heatMaps and compute loss
            err = 0
            loss = 0
            for j in range(num_channels-1):
                curr_heatMap = cv2.resize(heatMaps[:,:,j],(inputSizeNN,inputSizeNN))
                
                diff = np.subtract(curr_heatMap, labels[:,:,j])
                dot = np.power(diff, 2)
                loss += np.sum(dot)/(inputSizeNN*inputSizeNN)
                if (l == (len(layer_names)-1)):
                    x,y = findPoint(curr_heatMap)
                    err += np.sqrt(np.power(joints[j][0]-x,2)+np.power(joints[j][1]-y,2))
                plt.imshow(curr_heatMap)
                plt.waitforbuttonpress()
            loss_stage[l] += loss
        mpepj_model.append(float(err)/(num_channels-1))
    
    mpepj_value = np.mean(mpepj_model)
    val = dict([('iteration',[]), ('loss_iter',[]), ('loss_stage',[]), ('mpepj',[]), ('stage',[])])
    
    for l in range(len(layer_names)):
        val['iteration'].append(iterNumber)
        val['loss_iter'].append(np.sum(loss_stage))
        val['loss_stage'].append(float(loss_stage[l])/len(mpepj_model))
        val['mpepj'].append(mpepj_value)
        val['stage'].append(l+1)
        
    # TODO: check why result from Matlab is different from this one (close to 1 vs 0.02)
    return val
    
def combine_data(val, new_val):
    for i in range(len(new_val['iteration'])):
        val['iteration'].append(new_val['iteration'][i])
        val['loss_iter'].append(new_val['loss_iter'][i])
        val['loss_stage'].append(new_val['loss_stage'][i])
        val['mpepj'].append(new_val['mpepj'][i])
        val['stage'].append(new_val['stage'][i])
    return val

def getIter(item):
    regex_iteration = re.compile('pose_iter_(\d+).caffemodel')
    iter_match = regex_iteration.search(item)
    return int(iter_match.group(1))

def getLossOnValidationSet(json_file, models, mask_file):
    prototxt = models + 'pose_deploy.prototxt'
    files = [f for f in os.listdir(models) if f.endswith('.caffemodel')]
    files = sorted(files, key=getIter)
    val = dict([('iteration',[]), ('loss_iter',[]), ('loss_stage',[]), ('mpepj',[]), ('stage',[])])
    
    print 'Loading json file with annotations...'
    with open(json_file) as data_file:
        data_this = json.load(data_file)
        data_this = data_this['root']
        data = data_this
    print 'Done.'
    # index is in Matlab format
    print 'Loading mask file...'
    masks = sio.loadmat(mask_file)
    print 'Done.'
    
    numSample = len(data)
    print 'overall data %d' % len(data)
    idx = range(0, numSample, samplingRate)
    
    for i in range(len(files)):
        model_dir = '%s/%s' % (models, files[i])
        if (getIter(model_dir) < iter_start_from):
            continue
        new_val = runCaffeOnModel(data, model_dir, prototxt, idx, masks)
        val = combine_data(val, new_val)
    return val

def main():
    caffe.set_mode_gpu()
    caffe.set_device(device_id)
    
    caffe_dir = os.environ.get('CAFFE_HOME_CPM')
    json_file = '%s/models/cpm_architecture/jsonDatasets/H36M_annotations.json' % caffe_dir
    caffe_models_dir = '%s/models/cpm_architecture/prototxt/caffemodel/trial_5/' % caffe_dir
    mask_file = '%s/models/cpm_architecture/jsonDatasets/H36M_masks.mat' % caffe_dir
    output_file = '%svalidation_tmp.json' % caffe_models_dir
    
    loss = getLossOnValidationSet(json_file, caffe_models_dir, mask_file)
    
    with open(output_file, 'w+') as out:
        json.dump(loss, out)

if __name__ == '__main__':
    main()
