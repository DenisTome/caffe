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

def preprocessImages(NN, data, batch_size, batch_imgch, num_channels, masks):
    batch_info = np.empty((batch_size, 6),dtype=int)
    for b in range(batch_size):
        if (batch_size > 1):
            curr_data = data[b]
        else:
            curr_data = data
        
        img = ut.cv2.imread(curr_data['img_paths'])
        joints = np.array(curr_data['joint_self'])
        joints = ut.removeZ(joints)
        
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
            camera = masks['mask_camera'][0, fno - 1]
            action = masks['mask_action'][0, fno - 1]
            person = masks['mask_person'][0, fno - 1]
            metadata_ch = ut.generateMaskChannel(NN['inputSize'],
                                                 curr_data['annolist_index'], camera,
                                                 action, person)
            imgch = np.concatenate((resizedImage, center_channel[:,:,np.newaxis], metadata_ch), axis=2)
        
        imgch = np.transpose(imgch, (2, 0, 1))
        batch_imgch[b] = imgch
    return (batch_imgch, batch_info)

def postprocessHeatmaps(NN, batch_out, batch_data, batch_info, batch_size):
    loss = np.zeros(batch_size)
    err  = np.zeros(batch_size)
    for b in range(batch_size):
        if (batch_size > 1):
            curr_data = batch_data[b]
        else:
            curr_data = batch_data
        out = batch_out[b]
        heatMaps = ut.restoreSize(out, batch_info[b,:2], batch_info[b,2:])
        predictions = ut.findPredictions(NN['njoints'], heatMaps)
        gt = ut.filterJoints(ut.removeZ(curr_data['joint_self']))
        err[b] = ut.computeError(gt, predictions)
        labels = ut.generateHeatMaps(gt,  NN['outputSize'], NN['sigma'])
        labels = ut.restoreSize(labels, batch_info[b,:2], batch_info[b,2:])
        diff = np.subtract(heatMaps, labels).flatten()
        loss[b] = np.dot(diff,diff)/(NN['outputSize']*2*(NN['njoints']+1))
    return (err, loss)

def runCaffeOnModel(NN, net, data, masks, iteration, show_iter=10):
    # get just elements from the validation set
    val_idx = []
    for i in range(len(data)):
        if (not data[i]['isValidation']):
			continue
        val_idx.append(i)
    num_elem = len(val_idx)
    loss = np.zeros(num_elem)
    err  = np.zeros(num_elem)
    
    num_channels = ut.getNumChannelsLayer(net,'data')
    batch_size = ut.getBatchSizeLayer(net, 'data')
    p = 0
    i = 0
    while (i < num_elem):
        # TODO: not working show itertion
        # TODO: test why res is different
        if (np.mod(p, show_iter) == 0):
            print 'Model %r, Frame %d out of %d' % (iteration, i+1, len(val_idx))
            
        batch_imgch = np.empty((batch_size, num_channels, NN['inputSize'], NN['inputSize']))
        curr_batch_size = batch_size
        if ((i + batch_size) >= num_elem):
            # consider not perfect division of dataset size and batch_size
            curr_batch_size = num_elem - i
        
        if (curr_batch_size > 0):
            batch_data = data[val_idx[i:i+curr_batch_size]]
            (batch_imgch, batch_info) = preprocessImages(NN, batch_data, curr_batch_size,
                                                         batch_imgch, num_channels, masks)
        else:
            batch_data = data[val_idx[i]]
            (batch_imgch, batch_info) = preprocessImages(NN, batch_data, 1,
                                                         batch_imgch, num_channels, masks)
    
        ut.netForward(net, batch_imgch)
        layer_name = 'Mconv5_stage6_new'
        batch_out = ut.getOutputLayer(net, layer_name)
        
        if (curr_batch_size > 0):
            (b_err, b_loss) = postprocessHeatmaps(NN, batch_out, batch_data, batch_info, curr_batch_size)
            loss[i:i+curr_batch_size] = b_loss
            err[i:i+curr_batch_size] = b_err
        else:
            (b_err, b_loss) = postprocessHeatmaps(NN, batch_out, batch_data, batch_info, 1)
            loss[i] = b_loss
            err[i] = b_err
            
        i += batch_size
        p += 1
    
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
    for i in range(len(results)):
        x[i] = results[i]['iteration']
        mpjpes[i] = results[i]['mpjpe']
        losses[i] = results[i]['loss']
    ut.plt.subplot(211), ut.plt.plot(x,mpjpes,'ro-')
    ut.plt.title('mpjpe')
    ut.plt.xlabel('Iterations')
    ut.plt.grid()
    ut.plt.subplot(212), ut.plt.plot(x,losses,'bo-')
    ut.plt.title('loss')
    ut.plt.xlabel('Iterations')
    ut.plt.grid()
    ut.plt.suptitle('Errors on the validation set')
    ut.plt.show()

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


