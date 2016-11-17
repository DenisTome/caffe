# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""

import numpy as np
import caffe_utils as ut
import sys
import os

def getData():
    Dpath = '/home/denitome/PythonProjects/sfm_new/'
    json_file = ut.getCaffeCpm() + '/jsonDatasets/H36M_annotations_testSet.json'
    (data, num_elem) = ut.loadJsonFile(json_file)
    # 2d predictions    
    test = ut.sio.loadmat(Dpath+'Data/testSet_with_predictions.mat')
    mask_camera = test['mask_camera'][0]
    mask_action = test['mask_action'][0]
    preds = test['pred']
    # 3d predictions
    p_res_dict = ut.sio.loadmat(Dpath+'predicted_3d_poses_testset.mat')
    preds_3d = p_res_dict['pred_models']
    return (data, preds, preds_3d, mask_camera, mask_action)
    # mask person
    idx = np.where(mask_action == 2)[0]
    t = np.where((idx[:-1]+1) != idx[1:])[0][0] + 1
    fno_person = idx[t]
    mask_person = np.ones(num_elem)*9
    mask_person[fno_person:] = 11
    # all masks
    masks = {}
    masks['camera'] = mask_camera
    masks['action'] = mask_action
    masks['person'] = mask_person
    return (data, preds, preds_3d, masks)
    
def getIndex(masks, camera=1, person=9, action=2):
    idx = np.where((masks['camera'] == camera) & 
                   (masks['action'] == action) &
                   (masks['person'] == person))
    return idx[0]


(data, preds, preds_3d, masks) = getData()
idx = getIndex(masks)



    

    