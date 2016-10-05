# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:25:18 2016

@author: denitome
"""

import re
import json
import numpy as np
import caffe_utils as ut
import argparse

def runCaffeOnModel(NN, net, data, masks, iteration, show_iter=25):
    # get just elements from the validation set
    val_idx = []
    for i in range(len(data)):
        if (not data[i]['isValidation']):
			continue
        val_idx.append(i)

    loss = np.zeros(len(val_idx))
    err  = np.zeros(len(val_idx))
    
    for i in range(len(val_idx)):
        if (np.mod(i, show_iter) == 0):
            print 'Model %r, Iteration %d out of %d' % (iteration, i+1, len(val_idx))
            
        curr_data = data[val_idx[i]]
        img = ut.cv2.imread(curr_data['img_paths'])
        joints = np.array(curr_data['joint_self'])
        joints = ut.removeZ(joints)
        gt = joints.copy()
    
        center = ut.getCenterJoint(joints)
        img_width = img.shape[1]
        img_height = img.shape[0]
        box_points = ut.getBoundingBox(joints, center, NN['offset'], img_width, img_height)
        
        # manipulate image and joints for caffe
        (img_croppad, joints) = ut.cropImage(img, box_points, joints)
        (resizedImage, joints) = ut.resizeImage(img_croppad, NN['inputSize'], joints)
        resizedImage = np.divide(resizedImage, float(256))
        resizedImage = np.subtract(resizedImage, 0.5)
        
        # generate labels and center channel
        input_size = NN['inputSize']
        labels = ut.generateHeatMaps(ut.filterJoints(joints),  NN['outputSize'], NN['sigma'])
        center_channel = ut.generateGaussian(NN['sigma_center'], input_size, [input_size/2, input_size/2])
        
        num_channels = ut.getNumChannelsLayer(net, 'data')
        if (num_channels == 4):
            imgch = np.concatenate((resizedImage, center_channel[:,:,np.newaxis]), axis=2)
        else:
            fno = int(curr_data['annolist_index'])
            camera = masks['mask_camera'][0, fno - 1]
            action = masks['mask_action'][0, fno - 1]
            person = masks['mask_person'][0, fno - 1]
            metadata_ch = ut.generateMaskChannel(NN['inputSize'],
                                                 curr_data['annolist_index'], camera,
                                                 action, person)
            imgch = np.concatenate((resizedImage, center_channel[:,:,np.newaxis], metadata_ch), axis=2)

        imgch = np.transpose(imgch, (2, 0, 1))
        ut.netForward(net, imgch)
        
        # get results
        layer_name = 'Mconv5_stage6_new'
        out = ut.getOutputLayer(net, layer_name)
    
        heatMaps = ut.restoreSize(out, img.shape[:2], box_points)
        predictions = ut.findPredictions(NN['njoints'], heatMaps)
        
        gt = ut.filterJoints(gt)
        err[i] = ut.computeError(gt, predictions)
        labels = ut.restoreSize(labels, img.shape[:2], box_points)
        diff = np.subtract(heatMaps,labels).flatten()
        loss[i] = np.dot(diff,diff)/(NN['outputSize']*2*(NN['njoints']+1))
        
    val = {'iteration':iteration, 'mpjpe':err.mean(), 'loss':loss.mean()}
    return val    

def getIter(item):
    regex_iteration = re.compile('pose_iter_(\d+).caffemodel')
    iter_match = regex_iteration.search(item)
    return int(iter_match.group(1))

def getLossOnValidationSet(NN, models_dir, json_file, masks, prototxt, output, samplingRate):
    # get models
    files = [f for f in ut.os.listdir(models_dir) if f.endswith('.caffemodel')]
    files = sorted(files, key=getIter)
    
    print 'Loading json file with annotations...'
    (data, num_elem) = ut.loadJsonFile(json_file)
    print 'Loading mask file...'
    masks = ut.sio.loadmat(masks)
    print 'Done.'
    
    results = [dict() for x in range(len(files))]
    for i in range(len(files)):
        model_dir = '%s/%s' % (models_dir, files[i])
        iterNumber = getIter(model_dir)
        print '-------------------------------'
        print '  Evaluating iteration: %d' % iterNumber
        print '-------------------------------'
        net = ut.loadNetFromPath(model_dir, prototxt)
        val = runCaffeOnModel(NN, net, data[0:num_elem:samplingRate], masks, iterNumber)
        results[i] = val
        # save json
        with open(output, 'w+') as out:
            json.dump(results, out)
    return results

def checkModelsDir(path):
    if not ut.checkDirExists(path):
        raise Exception("Models directory does not exist")
    files = [f for f in ut.os.listdir(path) if f.endswith('.caffemodel')]
    if not len(files):
        raise Exception("No caffemodel file found in %r" % path)
    deploy_file = path+'/pose_deploy.prototxt'
    if not ut.checkFileExists(deploy_file):
        raise Exception("No prototxt file found in %r" % path)

def plotResults(results):
    if results is dict:
        print 'Iteration: %r\nMpjpe: %r\nLoss: %r' % (results['iteration'],results['mpjpe'],results['loss'])
        return
    x = np.empty(len(results))
    mpjpes = np.empty(len(results))
    losses = np.empty(len(results))
    for i in len(results):
        x[i] = results[i]['iteration']
        mpjpes[i] = results[i]['mpjpe']
        losses[i] = results[i]['loss']
    ut.plt.plot(x,mpjpes,'r',x,losses,'b')
    ut.plt.legend(('mpjpe','loss'))
    ut.plt.xlabel('Iterations')

def writeReadMe(path, json, masks, sampling):
    readMe_file = path + '/ReadMe.txt'
    with open(readMe_file,'w+') as readMe:
        readMe.write('Automatically generated ReadMe file\n\n')
        readMe.write('Validation run details:\n')
        readMe.write('Json file path: %r\n' % json)
        readMe.write('Maks file path: %r\n' % masks)
        readMe.write('Sampling: %r\n' % sampling)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir', metavar='models directory', type=str, help='Directory containing caffemodels and pose_deploy.prototxt')
    parser.add_argument('-s', dest='sampling', type=int, default=50, help='Sampling factor of the frames (default 50)')
    parser.add_argument('-o', dest='file_name', help='Name of the generated json file containing the results')
    parser.add_argument('--json', dest='json_file', type=str, help='Path to the json file containing the val-set data')
    parser.add_argument('--masks', dest='masks_file', type=str, help='Path to the mat file containing the val-set masks')
    parser.add_argument('--with_cpu', action='store_true', dest='with_cpu', help='Caffe uses CPU for feature extraction')
    
    args = parser.parse_args()
    checkModelsDir(args.models_dir)
    
    # setting up environment
    NN = ut.load_configuration(gpu=(not args.with_cpu))
    ut.setCaffeMode(NN['GPU'])
    
    # defining source files
    json_file = ut.getCaffeCpm()+'/jsonDatasets/H36M_annotations.json'
    if ((args.json_file is not None) and ut.checkFileExists(args.json_file)):
        json_file = args.json_file
    mask_file = ut.getCaffeCpm()+'/jsonDatasets/H36M_masks.mat'
    if ((args.masks_file is not None) and ut.checkFileExists(args.masks_file)):
        mask_file = args.masks_file
    # model definition
    prototxt = args.models_dir + '/pose_deploy.prototxt'
    # output file
    output_file = args.models_dir + '/validation.json'
    if args.file_name is not None:
        output_file = args.models_dir + '/' + args.file_name + '.json'
    results = getLossOnValidationSet(NN, args.models_dir, json_file,
                                     mask_file, prototxt, output_file, int(args.sampling))
    writeReadMe(args.models_dir, json_file, mask_file, int(args.sampling))
    
    plotResults(results)

if __name__ == '__main__':
    main()