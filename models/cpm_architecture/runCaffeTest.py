# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""

import numpy as np
import caffe_utils as ut
import argparse
import re

def preprocessImages(NN, data, batch_size, batch_imgch, num_channels):
    batch_info = np.empty((batch_size, 6),dtype=int)
    for b in range(batch_size):
        if (batch_size > 1):
            curr_data = data[b]
        else:
            curr_data = data
        
        img = ut.cv2.imread(curr_data['img_paths'])
        joints = np.array(curr_data['joint_self'])
        
        center = ut.getCenterJoint(joints)
        img_width = img.shape[1]
        img_height = img.shape[0]
        batch_info[b,:2] = img.shape[:2]
        box_points = ut.getBoundingBox(joints, center, NN['offset'], img_width, img_height)
        batch_info[b,2:] = box_points
        
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
        batch_imgch[b] = imgch
    return (batch_imgch, batch_info)
    
def postprocessHeatmaps(NN, batch_out, batch_data, batch_info, batch_size):
    pred = np.zeros((batch_size, NN['njoints']*2))
    err  = np.zeros(batch_size)
    for b in range(batch_size):
        if (batch_size > 1):
            curr_data = batch_data[b]
        else:
            curr_data = batch_data
        out = batch_out[b]
        heatMaps = ut.restoreSize(out, batch_info[b,:2], batch_info[b,2:])
        predictions = ut.findPredictions(NN['njoints'], heatMaps)
        pred[b] = predictions.flatten()
        gt = ut.filterJoints(curr_data['joint_self'])
        err[b] = ut.computeError(gt, predictions)
    return (pred, err)

def executeOnTestSet(NN, net, data, num_elem, offset=0, show_iter=20):
    preds  = np.empty((num_elem, NN['njoints']*2))
    errors = np.zeros(num_elem)
    frame_num = np.zeros(num_elem)
    
    num_channels = ut.getNumChannelsLayer(net,'data')
    batch_size = ut.getBatchSizeLayer(net, 'data')
    p = 0
    i = 0
    while (i < num_elem):
        if (np.mod(p, show_iter) == 0):
            print 'Frame %d out of %d' % (i+1, num_elem)
        batch_imgch = np.empty((batch_size, num_channels, NN['inputSize'], NN['inputSize']))
        curr_batch_size = batch_size
        if ((i + batch_size) >= num_elem):
            # consider not perfect division of dataset size and batch_size
            curr_batch_size = num_elem - i
        
        if (curr_batch_size > 0):
            batch_data = data[i:i+curr_batch_size]
            (batch_imgch, batch_info) = preprocessImages(NN, batch_data, curr_batch_size,
                                                         batch_imgch, num_channels)
        else:
            batch_data = data[i]
            (batch_imgch, batch_info) = preprocessImages(NN, batch_data, 1,
                                                         batch_imgch, num_channels)
                                                         
        ut.netForward(net, batch_imgch)
        layer_name = 'Mconv5_stage6_new'
        batch_out = ut.getOutputLayer(net, layer_name)
        
        if (curr_batch_size > 0):
            (b_preds, b_err) = postprocessHeatmaps(NN, batch_out, batch_data, batch_info, curr_batch_size)
            preds[i:i+curr_batch_size] = b_preds
            errors[i:i+curr_batch_size] = b_err
            frame_num[i:i+curr_batch_size] = i + offset
        else:
            (b_preds, b_err) = postprocessHeatmaps(NN, batch_out, batch_data, batch_info, 1)
            preds[i] = b_preds
            errors[i] = b_err
            frame_num[i] = i + offset
        i += batch_size
        p += 1
            
    return (preds, errors, frame_num)

def getIter(item):
    regex_iteration = re.compile('predictions_part(\d+).mat')
    iter_match = regex_iteration.search(item)
    return int(iter_match.group(1))

def mergeParts(NN, dir_path, num_elem, elems_per_part):
    files = [f for f in ut.os.listdir(dir_path) if f.endswith('.mat')]
    files = sorted(files, key=getIter)
    preds = np.empty((num_elem, NN['njoints']*2))
    errors = np.empty(num_elem)
    frame_num = np.empty(num_elem)
    for i in range(len(files)):
        part = getIter(files[i])
        curr_data = ut.sio.loadmat(dir_path+'/'+files[i])
        curr_data_size = curr_data['preds'].shape[0]
        start_idx = (part-1)*elems_per_part
        preds[start_idx:start_idx+curr_data_size] = curr_data['preds']
        errors[start_idx:start_idx+curr_data_size] = curr_data['errors']
        frame_num[start_idx:start_idx+curr_data_size] = curr_data['frame_num']
    return (preds, errors, frame_num)

def checkFilesExistance(prototxt, caffemodel):
    if not ut.checkFileExists(prototxt):
        raise Exception("Prototxt file not found at the given path: %r" % prototxt)
    if not ut.checkFileExists(caffemodel):
        raise Exception("Caffemodel file not found at the given path: %r" % caffemodel)

def getNameOutputFile(caffemodel):
    output_file = re.sub('.caffemodel', '_predictions.mat', caffemodel)
    return output_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prototxt', metavar='prototxt', type=str, help='Path to the prototxt file')
    parser.add_argument('caffemodel', metavar='caffemodel', type=str, help='Path to the caffemodel')
    parser.add_argument('-o', dest='out_directory', help='Output directory where to store the predictions')
    parser.add_argument('--json_file', dest='json_file', type=str, help='Path to the json file containing the test-set data')
    parser.add_argument('---with_cpu', action='store_true', dest='with_cpu', help='Caffe uses CPU for feature extraction')
    parser.add_argument('--num_parts', default=1, type=int, dest='num_parts', help='Split the test set into num_parts. Default set to 1')
    parser.add_argument('--run_part', default=1, type=int, dest='run_part', help='Run part of the splitted test set. Default set to 1')
    parser.add_argument('--merge_parts_dir', dest='merge_parts_dir', type=str, help='Merge part contained in the directory')
    
    args = parser.parse_args()
    checkFilesExistance(args.prototxt, args.caffemodel)
    
    # set environment
    NN = ut.load_configuration(gpu=(not args.with_cpu))
    ut.setCaffeMode(NN['GPU'])
    
    # get json file path
    json_file = ut.getCaffeCpm() + '/jsonDatasets/H36M_annotations_testSet.json'
    if args.json_file is not None:
        json_file = args.json_file
    
    # load caffe model
    net = ut.loadNetFromPath(args.caffemodel, args.prototxt)
    (data, num_elem) = ut.loadJsonFile(json_file)
    
    # Execution in multiple machines
    elems_per_part = int(np.floor(num_elem/int(args.num_parts)))
    offset_data = 0                                             # for not perfect divisions
    offset = elems_per_part*(int(args.run_part)-1)              # depending the part we are running
    if (args.run_part == args.num_parts):
        offset_data = num_elem - elems_per_part*int(args.num_part)
    idx_part = range(offset, offset + elems_per_part + offset_data)
    
    # Set output file path
    output_file = getNameOutputFile(args.caffemodel)
    if args.out_directory is not None:
        output_file = args.out_directory
        output_file += '/predictions.mat'

    if args.merge_parts_dir is not None:
        (preds, errors, frame_num) = mergeParts(NN, args.merge_parts_dir, num_elem, elems_per_part)
        ut.sio.savemat(output_file, {'preds':preds,'errors':errors,'frame_num':frame_num})
        return
    
    # run model on test set
    (preds, errors, frame_num) = executeOnTestSet(NN, net, data[idx_part], elems_per_part, offset)
    print 'Mean error on the test set: %r' % errors.mean()
    
    # save results
    if (args.num_part > 1):
        output_file += '/predictions_part%d.mat' % (int(args.run_part))
    ut.sio.savemat(output_file, {'preds':preds,'errors':errors,'frame_num':frame_num})

#if __name__ == '__main__':
#    main()

# TODO: test it again

NN = ut.load_configuration(gpu=True)
ut.setCaffeMode(NN['GPU'])
json_file = ut.getCaffeCpm() + '/jsonDatasets/H36M_annotations_testSet.json'
caffemodel = ut.getCaffeCpm() + '/prototxt/caffemodel/manifold_samearch3/pose_iter_110000.caffemodel'
prototxt = ut.getCaffeCpm() + '/prototxt/pose_deploy.prototxt'
net = ut.loadNetFromPath(caffemodel, prototxt)
(data, num_elem) = ut.loadJsonFile(json_file)
output_file = '/home/denitome/Dekstop/tmp.mat'
idx_part = range(0, 35)
(preds, errors, frame_num) = executeOnTestSet(NN, net, data[idx_part], 35, 0)



















