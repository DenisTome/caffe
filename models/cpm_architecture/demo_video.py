# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""

import numpy as np
import caffe_utils as ut
import os
import shutil
import subprocess

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

def framesOnSequence(data, preds, preds_3d, idx, output_dir):
    max_val = preds_3d[idx,-1].max()-preds_3d[idx,-1].min()
    save_idx = 0
    for frame in idx:
        print 'Frame %d of %d' % (save_idx, len(idx))
        img = ut.cv2.imread(data[frame]['img_paths'])
        img_2d_skel = ut.plotImageJoints(img, preds[frame], h=True)
        save_name = 'tmp_2d_frames/img_%d.png' % save_idx
        ut.plt.imsave(output_dir+save_name, img_2d_skel)
        
        save_name = 'tmp_3d_frames/img_%d.png' % save_idx
        ut.plot3DJoints_for_video(preds_3d[frame], save_img=output_dir+save_name, max_axis=max_val)
        ut.plt.close()
        
        save_idx += 1

def genVideo(output_dir, camera=1, action=2, person=9):
    output_dir = '%scamera_%d_action_%d_person_%d/' % (output_dir, camera, action, person)
    temp_2d = output_dir + 'tmp_2d_frames'
    temp_3d = output_dir + 'tmp_3d_frames'
    os.makedirs(output_dir)
    os.makedirs(temp_2d)
    os.makedirs(temp_3d)
    
    (data, preds, preds_3d, masks) = getData()
    idx = getIndex(masks, camera, person, action)
    
    # TODO: remove it
#    idx = idx[:20]
    
    # generate frames
    framesOnSequence(data, preds, preds_3d, idx, output_dir)
    
    # generate videos
    input_files = temp_2d+'/img_%d.png'
    output_file = output_dir+'/video_2d.mp4'
    subprocess.call(['ffmpeg', '-y', '-r', '10','-i', input_files, output_file])
    input_files = temp_3d+'/img_%d.png'
    output_file = output_dir+'/video_3d.mp4'
    subprocess.call(['ffmpeg', '-y', '-r', '10','-i', input_files, output_file])
    shutil.rmtree(temp_2d)
    shutil.rmtree(temp_3d)  
    
def getIndex(masks, camera, person, action):
    idx = np.where((masks['camera'] == camera) & 
                   (masks['action'] == action) &
                   (masks['person'] == person))
    return idx[0]


output_dir = '/home/denitome/Desktop/video/'
#genVideo(output_dir, camera=1, action=2, person=9)
#genVideo(output_dir, camera=1, action=3, person=9)
#genVideo(output_dir, camera=1, action=14, person=9)
genVideo(output_dir, camera=1, action=5, person=9)
genVideo(output_dir, camera=1, action=11, person=9)
#genVideo(output_dir, camera=1, action=12, person=9)
#genVideo(output_dir, camera=1, action=15, person=9)
#genVideo(output_dir, camera=1, action=16, person=9)

#frame=idx[34]
#camera=1
#action=2
#person=9


    

    