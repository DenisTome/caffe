# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:17:22 2016

@author: denis
"""



import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import os
import caffe
from scipy.stats import multivariate_normal
import subprocess
import glob

def load_frame_data():
    curr_data = dict([('img_paths',[]), ('img_height',[]), ('img_width',[]), ('joint_self',[]), ('objpos',[])])
    # load data from the correct path
    if os.uname()[1] == 'mac':
        curr_data['img_paths'] = '/data/7_2_1_1_1.jpg'
    else:
        curr_data['img_paths'] = '/data/Human3.6M/Data/S7/Images/7_2_1_1_1.jpg'   
        
    curr_data['img_height'] = 1002.0
    curr_data['img_width'] = 1000.0 
    curr_data['joint_self'] = [[478.825, 426.586, 1.0], [502.959, 426.908, 1.0], [492.314, 513.686, 1.0],
                               [504.287, 598.628, 1.0], [487.845, 596.459, 1.0], [482.328, 592.055, 1.0],
                               [454.16, 426.272, 1.0],  [453.088, 515.142, 1.0], [458.693, 602.433, 1.0],
                               [443.096, 601.427, 1.0], [434.982, 596.402, 1.0], [478.826, 426.566, 1.0],
                               [489.997, 385.009, 1.0], [488.261, 332.892, 1.0], [483.348, 320.655, 1.0],
                               [488.648, 300.006, 1.0], [488.261, 332.892, 1.0], [466.81, 347.095, 1.0],
                               [452.303, 402.205, 1.0], [438.008, 447.47, 1.0],  [438.008, 447.47, 1.0],
                               [432.889, 444.12, 1.0],  [431.13, 466.512, 1.0],  [431.13, 466.512, 1.0],
                               [488.261, 332.892, 1.0], [511.148, 346.596, 1.0], [522.583, 400.663, 1.0], 
                               [521.042, 444.943, 1.0], [521.042, 444.943, 1.0], [512.944, 439.956, 1.0],
                               [521.327, 470.664, 1.0], [521.327, 470.664, 1.0]]
    curr_data['objpos'] = [478.825, 426.586]
    return curr_data
    
def load_frame_data2():
    curr_data = dict([('img_paths',[]), ('img_height',[]), ('img_width',[]), ('joint_self',[]), ('objpos',[])])
    # load data from the correct path
    if os.uname()[1] == 'mac':
        curr_data['img_paths'] = '/data/7_2_1_1_1.jpg'
    else:
        curr_data['img_paths'] = '/data/Human3.6M/Data/S1/Images/1_2_1_1_451.jpg' 
        
    curr_data['img_height'] = 1002.0
    curr_data['img_width'] = 1000.0 
    curr_data['joint_self'] = [[528.382, 436.13, 1.0], [520.024, 429.862, 1.0], [519.54, 521.719, 1.0], [535.753, 608.809, 1.0], [506.126, 619.66, 1.0], [491.726, 620.034, 1.0], [537.13, 442.693, 1.0], [509.208, 533.766, 1.0], [535.445, 627.014, 1.0], [501.244, 632.528, 1.0], [485.257, 631.825, 1.0], [528.383, 436.109, 1.0], [541.069, 388.095, 1.0], [551.943, 334.586, 1.0], [541.667, 315.84, 1.0], [556.808, 297.474, 1.0], [551.943, 334.586, 1.0], [555.264, 340.277, 1.0], [514.595, 331.977, 1.0], [470.256, 317.536, 1.0], [470.256, 317.536, 1.0], [449.956, 308.317, 1.0], [465.932, 309.348, 1.0], [465.932, 309.348, 1.0], [551.943, 334.586, 1.0], [536.281, 345.834, 1.0], [480.172, 332.46, 1.0], [475.303, 293.627, 1.0], [475.303, 293.627, 1.0], [496.845, 290.377, 1.0], [470.583, 266.214, 1.0], [470.583, 266.214, 1.0]]
    curr_data['objpos'] = [528.382, 436.13]
    return curr_data

def load_frame_data3():
    curr_data = dict([('img_paths',[]), ('img_height',[]), ('img_width',[]), ('joint_self',[]), ('objpos',[])])
    # load data from the correct path
    if os.uname()[1] == 'mac':
        curr_data['img_paths'] = '/data/7_2_1_1_1.jpg'
    else:
        curr_data['img_paths'] = '/data/Human3.6M/Data/S1/Images/1_4_1_4_772.jpg'
        
    curr_data['img_height'] = 1002.0
    curr_data['img_width'] = 1000.0 
    curr_data['joint_self'] = [[506.122, 410.273, 1.0], [474.839, 408.803, 1.0], [531.422, 482.896, 1.0], [511.176, 578.844, 1.0], [533.419, 614.59, 1.0], [542.036, 620.251, 1.0], [536.481, 411.722, 1.0], [582.798, 480.397, 1.0], [546.8, 572.368, 1.0], [557.26, 609.143, 1.0], [567.418, 616.684, 1.0], [506.127, 410.247, 1.0], [507.211, 349.987, 1.0], [517.139, 282.33, 1.0], [516.949, 260.902, 1.0], [513.727, 231.547, 1.0], [517.139, 282.33, 1.0], [548.143, 306.372, 1.0], [579.232, 368.856, 1.0], [580.02, 350.102, 1.0], [580.02, 350.102, 1.0], [566.154, 324.815, 1.0], [583.084, 348.59, 1.0], [583.084, 348.59, 1.0], [517.139, 282.33, 1.0], [482.832, 291.701, 1.0], [461.248, 274.452, 1.0], [531.674, 269.889, 1.0], [531.674, 269.889, 1.0], [543.879, 269.143, 1.0], [568.453, 279.164, 1.0], [568.453, 279.164, 1.0]]
    curr_data['objpos'] = [506.122, 410.273]
    return curr_data

def load_frame_data4():
    curr_data = dict([('img_paths',[]), ('img_height',[]), ('img_width',[]), ('joint_self',[]), ('objpos',[])])
    # load data from the correct path
    if os.uname()[1] == 'mac':
        curr_data['img_paths'] = '/data/7_2_1_1_1.jpg'
    else:
        curr_data['img_paths'] = '/data/Human3.6M/Data/S1/Images/1_10_2_2_2245.jpg'
        
    curr_data['img_height'] = 1000.0
    curr_data['img_width'] = 1000.0 
    curr_data['joint_self'] = [[564.189, 542.17, 1.0], [539.354, 533.787, 1.0], [523.676, 532.228, 1.0], [535.791, 578.993, 1.0], [507.485, 583.817, 1.0], [494.835, 584.395, 1.0], [589.255, 550.626, 1.0], [559.812, 517.437, 1.0], [568.333, 586.543, 1.0], [562.044, 597.598, 1.0], [558.872, 599.93, 1.0], [564.196, 542.152, 1.0], [568.592, 500.814, 1.0], [577.51, 471.886, 1.0], [568.549, 469.813, 1.0], [576.51, 446.029, 1.0], [577.51, 471.886, 1.0], [604.259, 485.076, 1.0], [625.065, 520.473, 1.0], [639.746, 552.754, 1.0], [639.746, 552.754, 1.0], [620.314, 556.712, 1.0], [643.808, 571.146, 1.0], [643.808, 571.146, 1.0], [577.51, 471.886, 1.0], [546.208, 476.744, 1.0], [497.731, 509.686, 1.0], [476.559, 558.211, 1.0], [476.559, 558.211, 1.0], [494.157, 564.407, 1.0], [462.987, 582.936, 1.0], [462.987, 582.936, 1.0]]
    curr_data['objpos'] = [564.189, 542.17]
    return curr_data

def load_frame_data5():
    curr_data = dict([('img_paths',[]), ('img_height',[]), ('img_width',[]), ('joint_self',[]), ('objpos',[])])
    # load data from the correct path
    if os.uname()[1] == 'mac':
        curr_data['img_paths'] = '/data/7_2_1_1_1.jpg'
    else:
        curr_data['img_paths'] = '/data/Human3.6M/Data/S8/Images/8_11_1_4_2695.jpg'
        
    curr_data['img_height'] = 1002.0
    curr_data['img_width'] = 1000.0 
    curr_data['joint_self'] = [[499.245, 423.753, 1.0], [467.188, 420.581, 1.0], [499.79, 465.638, 1.0], [490.542, 565.647, 1.0], [504.705, 586.105, 1.0], [509.594, 587.359, 1.0], [530.71, 426.887, 1.0], [571.377, 469.525, 1.0], [538.899, 568.547, 1.0], [546.16, 588.265, 1.0], [550.97, 583.713, 1.0], [499.245, 423.731, 1.0], [501.728, 364.585, 1.0], [503.998, 310.992, 1.0], [508.635, 293.681, 1.0], [506.142, 266.662, 1.0], [503.998, 310.992, 1.0], [540.609, 323.976, 1.0], [559.6, 388.039, 1.0], [504.389, 388.063, 1.0], [504.389, 388.063, 1.0], [499.599, 364.974, 1.0], [483.107, 393.249, 1.0], [483.107, 393.249, 1.0], [503.998, 310.992, 1.0], [466.858, 324.306, 1.0], [420.172, 373.07, 1.0], [380.758, 362.984, 1.0], [380.758, 362.984, 1.0], [396.184, 355.135, 1.0], [361.707, 335.688, 1.0], [361.707, 335.688, 1.0]]
    curr_data['objpos'] = [499.245, 423.753]
    return curr_data

def load_configuration(gpu = True):
    NN = dict()
    NN['offset'] = 25
    NN['samplingRate'] = 5
    NN['offset'] = 25
    NN['inputSize'] = 368
    NN['outputSize'] = 46
    NN['njoints'] = 17
    NN['sigma'] = 7
    NN['sigma_center'] = 21
    NN['stride'] = 8
    NN['GPU'] = gpu
    return NN

def loadNet(dir_name, file_detail):
    # defining model
    caffe_home = os.environ['CAFFE_HOME_CPM']
    def_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/pose_deploy.prototxt' % (caffe_home, dir_name)
    if isinstance(file_detail, basestring):
        model_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/%s.caffemodel' % (caffe_home, dir_name, file_detail)
    else:
        model_file = '%s/models/cpm_architecture/prototxt/caffemodel/%s/pose_iter_%d.caffemodel' % (caffe_home, dir_name, file_detail)
    net = caffe.Net(def_file, model_file, caffe.TEST)
    return net
    

def filterJoints(joints_orig):
    joints_idx = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    joints = [0] * len(joints_idx)
    for j in range(len(joints_idx)):
        joints[j] = map(int, joints_orig[joints_idx[j]])
    return joints
    
def getBoundingBox(NN, data):
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
    offset_x = max_x + NN['offset']
    offset_y = max_y + NN['offset']
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
    
def findPoint(heatMap):
    idx = np.where(heatMap == heatMap.max())
    x = idx[1][0]
    y = idx[0][0]
    return x,y

def generateGaussian(NN, position, center_map = False):
    if (center_map):
        sigma_sq = np.power(NN['sigma_center'],2)
    else:
        sigma_sq = np.power(NN['sigma'],2)
        
    # build covariance matrix
    Sigma = [[sigma_sq,0],[0,sigma_sq]]
    x, y = np.mgrid[0:NN['inputSize'], 0:NN['inputSize']]
    pos = np.dstack((x, y))
    rv = multivariate_normal([position[1],position[0]], Sigma)
    
    tmp = rv.pdf(pos)
    hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
    return hmap

def mask_from_file(file_name):
    while (file_name.find('/') >= 0):
        file_name = file_name[file_name.find('/')+1:]
    file_name = file_name[:file_name.find('.jpg')]
    data = file_name.split('_')
    
    person = data[0]
    action = data[1]
    camera = data[3]
    fno = data[4]
    return (fno, camera, action, person)
    

def preprocess_image(NN, curr_data):
    # read the image
    img = cv2.imread(curr_data['img_paths'])
    
    center = map(int, curr_data['objpos'])
    bbox = getBoundingBox(NN, curr_data)
    
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
        
    resizedImage = cv2.resize(img_croppad, (NN['inputSize'],NN['inputSize']), interpolation = cv2.INTER_CUBIC)
    fx = float(NN['inputSize'])/img_croppad.shape[1]
    fy = float(NN['inputSize'])/img_croppad.shape[0]
    center = map(int, np.multiply(center, [fx,fy]))
    for j in range(len(joints)):
        joints[j] = map(int, np.multiply(joints[j], [fx,fy]))
    
    return (resizedImage, joints, center)
    
def caffe_data(NN, net, img, center, data):
    NN['inputSize'] = net.blobs['data'].width
    num_channels = net.blobs['data'].channels

    # generate center heat-map
    center_hm = np.zeros((NN['inputSize'],NN['inputSize'],1))
    center_hm[:,:,0] = generateGaussian(NN, center, center_map=True)
    
    # generate channel for metadata
    (fno, camera, action, person) = mask_from_file(data['img_paths'])
    metadata_channel = np.zeros((NN['inputSize'],NN['inputSize'],1))
    metadata_channel[0,0,0] = fno
    metadata_channel[0,1,0] = camera
    metadata_channel[0,2,0] = action
    metadata_channel[0,3,0] = person
    
    input_img = np.divide(img, float(256))
    input_img = np.subtract(input_img, 0.5)
    
    if num_channels == 4:
        img4ch = np.concatenate((input_img, center_hm), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))
        # Give input
        net.blobs['data'].data[...] = img4ch    
    else:
        img5ch = np.concatenate((input_img, center_hm, metadata_channel), axis=2)
        img5ch = np.transpose(img5ch, (2, 0, 1))
        # Give input
        net.blobs['data'].data[...] = img5ch
    net.forward()

# Load data and configuration
NN = load_configuration(gpu = False)
data = load_frame_data3()
#
## take caffe input data
(NN_img, joints, center) = preprocess_image(NN, data)
if NN['GPU']:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

# Load caffe model
net = loadNet('manifold_initialised', 'initialisation_zero')
net2 = loadNet('trial_5', 50000)
caffe_data(NN, net, NN_img, center, data)
caffe_data(NN, net2, NN_img, center, data)

## Get outputs
layer_names = ['conv7_stage1_new', 'Mconv5_stage2_new', 'Mconv5_stage3_new',
               'Mconv5_stage4_new', 'Mconv5_stage5_new', 'Mconv5_stage6_new']
manifold_names = ['manifolds_stage1','manifolds_stage2','manifolds_stage3',
                  'manifolds_stage4','manifolds_stage5']

idx = 3
labels = net.blobs.get(layer_names[idx]).data
labels = np.reshape(labels,(18,NN['outputSize'],NN['outputSize']))
labels = np.transpose(labels, (1, 2, 0))
labels2 = net2.blobs.get(layer_names[idx]).data
labels2 = np.reshape(labels2,(18,NN['outputSize'],NN['outputSize']))
labels2 = np.transpose(labels2, (1, 2, 0))
manifold = net.blobs.get(manifold_names[idx]).data
    
c_dir = os.getcwd()
if not os.path.exists("tmp"):
    os.mkdir('tmp')
fig = plt.figure()
for h in range(NN['njoints'] + 1):
    plt.subplot(141), plt.title('img')
    plt.imshow(NN_img)
    plt.subplot(142), plt.title('old model')
    plt.imshow(labels2[:,:,h])
    plt.subplot(143), plt.title('new model')
    plt.imshow(labels[:,:,h])
    plt.subplot(144), plt.title('manifold')
    plt.imshow(manifold[0,h])
    plt.draw()
#    plt.waitforbuttonpress()
    plt.savefig('tmp/h%02d.png' % h)

os.chdir('tmp')
subprocess.call([
    'ffmpeg', '-framerate', '1', '-i', 'h%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'sample2.mp4'
])
for file_name in glob.glob("*.png"):
    os.remove(file_name)
os.chdir(c_dir)

    

layer_names_new = list(net._layer_names)
layer_names_old = list(net2._layer_names)

idx_new = 36
idx_old = 33
output_layer_net     = net.blobs.get(layer_names_new[idx_new]).data
output_layer_net_old = net2.blobs.get(layer_names_old[idx_old]).data
print 'The same: %r' % np.array_equal(output_layer_net, output_layer_net_old)
plt.subplot(121), plt.imshow(output_layer_net_old[0][0])
plt.subplot(122),plt.imshow(output_layer_net_old[0][0])
#
#idx = 110
#np.abs(np.subtract(output_layer_net[0,idx],output_layer_net_old[0,idx])).sum()
#
#tmp = output_layer_net[0][1]
#tmp1 = output_layer_net_old[0][1]
#np.array_equal(tmp,tmp1)

#img = plt.imread(data[1050450]['img_paths'])
#plt.imshow(img)