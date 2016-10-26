# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""

import numpy as np
import caffe_utils as ut
import re
import sys
import os
lib_path = os.environ['CAFFE_HOME_CPM']
sys.path.insert(0,'%s/python/manifold/' % lib_path)
import upright_cam as uc

def load_parameters():
    """Load trained parameters and camera rotation matrices"""
    Dpath = '%s/models/cpm_architecture/data/' % lib_path
    par   = ut.sio.loadmat(Dpath+'camera_param.mat')
    train = ut.sio.loadmat(Dpath + 'train_basis_allactions.mat')
    R    = np.empty((4,3,3))
    R[0] = par['c1'][0,0]['R']
    R[1] = par['c2'][0,0]['R']
    R[2] = par['c3'][0,0]['R']
    R[3] = par['c4'][0,0]['R']
    return (R, train['e'], train['z'], train['a'].std(0)**-1 )
    
def normalise_data(w):
    """Normalise data by centering all the 2d skeletons (0,0) and by rescaling
    all of them such that the height is 2.0 (expressed in meters)."""
    
    def centre(m):
        """Returns centered data in position (0,0)"""
        return ((m.T - m.mean(1)).T, m.mean(1))
        
    w = w.reshape(-1,2).transpose(1,0)
    (d2, mean) = centre(w)
    m2 = d2[1,:].min(0)/2.0
    m2 = m2-d2[1,:].max(0)/2.0
    crap = (m2 == 0)
    if crap:
        m2 = 1.0
    d2 /= m2
    return (d2, m2, mean)
        
def preprocessImage(NN, curr_data, batch_size, num_channels):
    info = np.empty(6, dtype=int)
        
    img = ut.cv2.imread(curr_data['img_paths'])
    joints = np.array(curr_data['joint_self'])
    
    center = ut.getCenterJoint(joints)
    img_width = img.shape[1]
    img_height = img.shape[0]
    info[:2] = img.shape[:2]
    box_points = ut.getBoundingBox(joints, center, NN['offset'], img_width, img_height)
    info[2:] = box_points
    
    # manipulate image and joints for caffe
    (img_croppad, joints) = ut.cropImage(img, box_points, joints)
    (resizedImage, joints) = ut.resizeImage(img_croppad, NN['inputSize'], joints)
    resizedImage = np.divide(resizedImage, float(256))
    resizedImage = np.subtract(resizedImage, 0.5)
    
    # generate labels and center channel
    input_size = NN['inputSize']
    center_channel = ut.generateGaussian(NN['sigma_center'], input_size, [input_size/2, input_size/2])
    
    if (num_channels == 4):
        imgch = np.concatenate((resizedImage, center_channel[:,:,np.newaxis]), axis=2)
    else:
        fno = int(curr_data['annolist_index'])
        camera = curr_data['camera']
        action = curr_data['action']
        person = curr_data['person']
        metadata_ch = ut.generateMaskChannel(NN['inputSize'], fno, camera, action, person)
        imgch = np.concatenate((resizedImage, center_channel[:,:,np.newaxis], metadata_ch), axis=2)
    
    imgch = np.transpose(imgch, (2, 0, 1))
    return (imgch, info)

def restoreSize(channels, channels_size, box_points):
    """Given the channel, it resize and place the channel in the right position
    in order to have a final estimation in the original image coordinate system.
    Channel has the format (c x w x h) """
    assert(channels.ndim == 3)
    num_channels = channels.shape[0]
    channels = channels.transpose(1,2,0)
    new_img = ut.cv2.resize(channels, (box_points[2],box_points[3]), interpolation = ut.cv2.INTER_LANCZOS4)

    reecreated_img = np.zeros((channels_size[0], channels_size[1], num_channels))
    reecreated_img[:,:,-1] = np.ones((channels_size[0], channels_size[1]))
    reecreated_img[box_points[1]:box_points[1]+box_points[3],box_points[0]:box_points[0]+box_points[2]] = new_img
    return reecreated_img
    
def postprocessHeatmaps(NN, out, curr_data, info):
    heatMaps = restoreSize(out, info[:2], info[2:])
    predictions = ut.findPredictions(NN['njoints'], heatMaps)
    gt = ut.filterJoints(curr_data['joint_self'])
    err = ut.computeError(gt, predictions)
    return (predictions, heatMaps, err)

def executeOnFrame(NN, net, data, output_dir, astr=False):
    num_channels = ut.getNumChannelsLayer(net,'data')
    joints = np.array(data['joint_self'])
    
    # Get Image
    img_orig = ut.cv2.imread(data['img_paths'])
    box_points = ut.getBoundingBox(joints, ut.getCenterJoint(joints),
                               75, img_orig.shape[1], img_orig.shape[0])
    img = ut.cropImage(img_orig, box_points)
    save_name = output_dir + 'image.png'
    if astr:
        save_name = output_dir + 'image' + astr + '.png'
    ut.plt.imsave(save_name, img)
    
    
    # Extract 2D predictions
    (imgch, info) = preprocessImage(NN, data, 1, num_channels)
    ut.netForward(net, imgch)
    img = ut.convertImgCv2(img)
    layer_name = 'Mconv5_stage6_new'
    out = ut.getOutputLayer(net, layer_name)
    (pred, heatMaps, err) = postprocessHeatmaps(NN, out, data, info)
    img_2d_skel = ut.plotImageJoints(img_orig, pred, h=True)
    img_2d_skel = ut.cropImage(img_2d_skel, box_points)
    save_name = output_dir + 'image_2d.png'
    if astr:
        save_name = output_dir + 'image_2d' + astr + '.png'
    ut.plt.imsave(save_name, img_2d_skel)
    
    # Plot data
    heatMap = ut.cropImage(heatMaps[:,:,-1], box_points)
    save_name = output_dir + 'image_hm.png'
    if astr:
        save_name = output_dir + 'image_hm' + astr + '.png'
    ut.plt.imsave(save_name, heatMap)
    
    Lambda = 0.05
    (default_r, e, z, weights) = load_parameters()
    (w, s, mean) = normalise_data(pred.flatten())
    w = w[np.newaxis,:]
    (a, r) = uc.estimate_a_and_r(w, e, z, default_r[int(data['camera'])], Lambda*weights)
    for j in xrange(10):
        r = uc.reestimate_r(w, z, a, e, default_r[int(data['camera'])], r)
        (a, res) = uc.reestimate_a(w, e, z, r, default_r[int(data['camera'])], Lambda*weights)
    mod = uc.build_model(a, e, z).squeeze()
    save_name = output_dir + 'image_3d.pdf'
    if astr:
        save_name = output_dir + 'image_3d' + astr + '.pdf'
    ut.plot3DJoints(-mod, save_pdf=save_name)
    
    return (img, heatMap, img_2d_skel)

def checkFilesExistance(prototxt, caffemodel):
    if not ut.checkFileExists(prototxt):
        raise Exception("Prototxt file not found at the given path: %r" % prototxt)
    if not ut.checkFileExists(caffemodel):
        raise Exception("Caffemodel file not found at the given path: %r" % caffemodel)

def getNameOutputFile(caffemodel):
    output_file = re.sub('\.caffemodel', '_predictions.mat', caffemodel)
    return output_file
    
def getIndex(data, camera=1, person=7, action=1, fno=0):
    action += 1 # action 1 is skept
    for i in range(len(data)):
        if (data['camera']==camera and data['person']==person and data['action']==action):
            break
    return i+fno

## SET ENVIRONMENT

caffemodel = ut.getCaffeCpm() + '/prototxt/caffemodel/manifold_diffarch3/to_test/pose_iter_16000.caffemodel'
prototxt = ut.getCaffeCpm() + '/prototxt/pose_deploy_singleimg.prototxt'
output_dir = '/home/denitome/Desktop/'
checkFilesExistance(prototxt, caffemodel)

# TODO: change it back
# NN = ut.load_configuration(gpu=(not args.with_cpu))
NN = ut.load_configuration(gpu=False)
ut.setCaffeMode(NN['GPU'])

# load caffe model
json_file = ut.getCaffeCpm() + '/jsonDatasets/H36M_annotations_testSet.json'
net = ut.loadNetFromPath(caffemodel, prototxt)
(data, num_elem) = ut.loadJsonFile(json_file)

executeOnFrame(NN, net, data, output_dir)