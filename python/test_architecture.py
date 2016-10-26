import os
import numpy as np
import sys
lib_path = os.environ['CAFFE_HOME_CPM']
sys.path.insert(0,'%s/models/cpm_architecture/' % lib_path)

import caffe_utils as ut
   
def produceInput(NN):
    json_file = ut.getCaffeCpm() + '/jsonDatasets/H36M_annotations_testSet.json'
    (data, num_elem) = ut.loadJsonFile(json_file)
    curr_data = data[0]
    
    img = ut.cv2.imread(curr_data['img_paths'])
    joints = np.array(curr_data['joint_self'])
    
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
    center_channel = ut.generateGaussian(NN['sigma_center'], input_size, [input_size/2, input_size/2])
    
    fno = int(curr_data['annolist_index'])
    camera = curr_data['camera']
    action = curr_data['action']
    person = curr_data['person']
    metadata_ch = ut.generateMaskChannel(NN['inputSize'], fno, camera, action, person)
    imgch = np.concatenate((resizedImage, center_channel[:,:,np.newaxis], metadata_ch), axis=2)
    
    imgch = np.transpose(imgch, (2, 0, 1))
    return imgch


NN = ut.load_configuration(gpu=False)
ut.setCaffeMode(NN['GPU'])
imgch = produceInput(NN)

net_orig = ut.loadNetFromPath('tmp/pose_iter_110000.caffemodel', 'tmp/pose_deploy_1.prototxt')
net_new  = ut.loadNetFromPath('tmp/pose_iter_110000.caffemodel', 'tmp/pose_deploy_2.prototxt')

ut.netForward(net_orig, imgch)
ut.netForward(net_new, imgch)

res_orig = ut.getOutputLayer(net_orig, 'conv7_stage1_new')
res_new  = ut.getOutputLayer(net_new, 'conv7_stage1_new')
assert(np.array_equal(res_orig,res_new))

res_orig = ut.getOutputLayer(net_orig, 'conv7_stage1_new')
res_new  = ut.getOutputLayer(net_new, 'merge_hm_stage1')
assert(np.array_equal(res_orig,res_new))

res_orig = ut.getOutputLayer(net_orig, 'Mconv5_stage2_new')
res_new  = ut.getOutputLayer(net_new, 'Mconv5_stage2_new')
assert(np.array_equal(res_orig,res_new))

res_orig = ut.getOutputLayer(net_orig, 'Mconv5_stage2_new')
res_new  = ut.getOutputLayer(net_new, 'merge_hm_stage2')
assert(np.array_equal(res_orig,res_new))

res_orig = ut.getOutputLayer(net_orig, 'Mconv5_stage4_new')
res_new  = ut.getOutputLayer(net_new, 'Mconv5_stage4_new')
assert(np.array_equal(res_orig,res_new))

assert(ut.getParamLayer(net_new, 'merge_hm_stage4')[0] == 0.5)

res_orig = ut.getOutputLayer(net_orig, 'manifolds_stage4')
res_new  = ut.getOutputLayer(net_new, 'merge_hm_stage4')
assert(np.allclose(res_orig,res_new, rtol=1e-6, atol=0))

#net_orig.blobs.get('conv6_stage1').data.shape
#net_orig.blobs.get('conv7_stage1_new').diff.shape
#net_orig.params.get('conv7_stage1_new')[0].data.shape