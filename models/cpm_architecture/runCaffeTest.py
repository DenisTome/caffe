# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""

import numpy as np
import caffe_utils as ut
import argparse
import re

def predictionsFromFrame(NN, net, curr_data):
    # read data
    img = ut.cv2.imread(curr_data['img_paths'])
    joints = np.array(curr_data['joint_self'])
    
    center = ut.getCenterJoint(joints)
    img_width = img.shape[1]
    img_height = img.shape[0]
    box_points = ut.getBoundingBox(joints, center, NN['offset'], img_width, img_height)
    
    # crop around person
    img_croppad = ut.cropImage(img, box_points)
    resizedImage = ut.resizeImage(img_croppad, NN['inputSize'])
    resizedImage = np.divide(resizedImage, float(256))
    resizedImage = np.subtract(resizedImage, 0.5)
    
    sigma = NN['sigma_center']
    input_size = NN['inputSize']
    center_channel = ut.generateGaussian(sigma, input_size, [input_size/2, input_size/2])
    
    num_channels = ut.getNumChannelsLayer(net, 'data')
    if (num_channels == 4):
        imgch = np.concatenate((resizedImage, center_channel[:,:,np.newaxis]), axis=2)
    else:
        metadata_ch = ut.generateMaskChannel(NN['inputSize'],
                                             curr_data['annolist_index'], curr_data['camera'],
                                             curr_data['action'], curr_data['person'])
        imgch = np.concatenate((resizedImage, center_channel[:,:,np.newaxis], metadata_ch), axis=2)
        
    imgch = np.transpose(imgch, (2, 0, 1))
    ut.netForward(net, imgch)
    
    layer_name = 'Mconv5_stage6_new'
    out = ut.getOutputLayer(net, layer_name)
    
    heatMaps = ut.restoreSize(out, img.shape[:2], box_points)
    predictions = ut.findPredictions(NN['njoints'], heatMaps)
    # prob not here
    gt = ut.filterJoints(joints)
    err = np.sqrt(np.power(gt-predictions,2).sum(1)).mean()
    return (predictions, err)

def executeOnTestSet(NN, net, data, num_elem, offset=0, show_iter=50):
    preds  = np.empty((num_elem, NN['njoints']*2))
    errors = np.zeros(num_elem)
    frame_num = np.zeros(num_elem)
    for fno in range(num_elem):
        if not np.mod(fno, show_iter):
            print 'Frame %r of %r' % (fno, num_elem)
        curr_data = data[fno]
        (curr_pred, err) = predictionsFromFrame(NN, net, curr_data)
        preds[fno] = curr_pred.flatten()
        errors[fno] = err
        frame_num[fno] = fno + offset
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
    offset_data = 0 # for not perfect divisions
    offset = elems_per_part*(int(args.run_part)-1)
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




















