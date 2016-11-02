# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:38:16 2016

@author: denitome
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import os
import caffe
from scipy.stats import multivariate_normal
import json

def load_configuration(offset=25, samplingRate=5, inputSize=368, outputSize=46,
                       njoints=17, sigma=7, sigma_center=21, stride=8, gpu=True):
    """Set-up default configurations"""
    NN = dict()
    NN['offset'] = offset
    NN['samplingRate'] = samplingRate
    NN['inputSize'] = inputSize
    NN['outputSize'] = outputSize
    NN['njoints'] = njoints
    NN['sigma'] = sigma
    NN['sigma_center'] = sigma_center
    NN['stride'] = stride
    NN['GPU'] = gpu
    return NN

def loadNet(folder_name, file_detail):
    """Load caffe model"""
    # defining models path
    caffe_home = os.environ['CAFFE_HOME_CPM']
    sub_dir = '%s/models/cpm_architecture/prototxt/caffemodel' % caffe_home
    # set file paths
    def_file = '%s/%s/pose_deploy.prototxt' % (sub_dir,folder_name)
    if isinstance(file_detail, basestring):
        model_file = '%s/%s/%s.caffemodel' % (sub_dir, folder_name, file_detail)
    else:
        model_file = '%s/%s/pose_iter_%d.caffemodel' % (sub_dir, folder_name, file_detail)
    # load caffe model
    net = caffe.Net(def_file, model_file, caffe.TEST)
    return net

def loadNetFromPath(caffemodel, prototxt):
    """Load caffe model"""
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return net

def setCaffeMode(gpu, device = 0):
    """Initialise caffe"""
    if gpu:
        caffe.set_mode_gpu()
        caffe.set_device(device)
    else:
        caffe.set_mode_cpu()
    
def getCenterJoint(joints):
    if not checkJointsNonLinearised(joints):
        joints = xyJoints(np.array(joints))
    return joints[0]

def getBoundingBox(joints, center, offset, img_width, img_height):
    """Get bounding box containing all joints keeping as constant as possible
    the aspect ratio of the box"""
    if not checkJointsNonLinearised(joints):
        joints = xyJoints(np.array(joints))
    max_x = -1
    max_y = -1
    for i in range(joints.shape[0]):
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
    
    box_points = np.empty(4)
    # pos - 1 because joints are expressed in Matlab format
    box_points[:2] = center-[offset_x,offset_y]-1
    box_points[2:] = np.multiply([offset_x,offset_y],2)
    # check that are inside the image
    if (box_points[0] + box_points[2] > img_width):
        box_points[2] = img_width - box_points[0]
    if (box_points[0] < 0):
        box_points[0] = 0
    if (box_points[1] + box_points[3] > img_height):
        box_points[3] = img_height - box_points[1]
    if (box_points[1] < 0):
        box_points[1] = 0
        
    return np.round(box_points).astype(int)

def findPoint(heatMap):
    """Find maximum point in a given heat-map"""
    idx = np.where(heatMap == heatMap.max())
    x = idx[1][0]
    y = idx[0][0]
    return x,y

def generateGaussian(Sigma, input_size, position):
    """Generate gaussian in the specified position with the specified sigma value"""
    
    if not isinstance(Sigma,(np.ndarray, list)):
        # build covariance matrix
        sigma_sq = np.power(Sigma, 2)
        Sigma = np.eye(2)*sigma_sq      

    x, y = np.mgrid[0:input_size, 0:input_size]
    pos = np.dstack((x, y))
    rv = multivariate_normal([position[1], position[0]], Sigma)
    
    tmp = rv.pdf(pos)
    hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
    idx = np.where(hmap.flatten() <= np.exp(-4.6052))
    hmap.flatten()[idx] = 0
    return hmap

def generateHeatMaps(joints, input_size, sigma):
    """Generate heat-map for each of the joints and one overal heat-maps containing
    all the joints positions."""
    heatMaps = np.zeros((joints.shape[0]+1, input_size, input_size))
    for i in range(joints.shape[0]):
        heatMaps[i] = generateGaussian(sigma, input_size, joints[i])
    # generating last heat maps which contains all joint positions
    heatMaps[-1] = np.maximum(1.0-heatMaps[0:heatMaps.shape[0]-1].max(axis=0), 0)
    return heatMaps

def cropImage(image, box_points, joints=False):
    """Given a box with the format [x_min, y_min, width, height]
    it returnes the cropped image"""
    croppedImage = image[box_points[1]:box_points[1]+box_points[3],box_points[0]:box_points[0]+box_points[2]]
    if joints is False:
        return croppedImage
    if not checkJointsNonLinearised(joints):
        joints = xyJoints(np.array(joints))
    for j in range(joints.shape[0]):
        joints[j][0] -= box_points[0]
        joints[j][1] -= box_points[1]
    return (croppedImage, joints)

def resizeImage(image, new_size, joints=False):
    """Resize image to a defined size. The size if the same for width and height
    If joint positions are provided, they are rescaled as well."""
    resizedImage = cv2.resize(image, (new_size, new_size), interpolation = cv2.INTER_CUBIC)
    if joints is False:
        return resizedImage
    if not checkJointsNonLinearised(joints):
        joints = xyJoints(np.array(joints))
    fx = float(new_size)/image.shape[1]
    fy = float(new_size)/image.shape[0]
    assert(fx != 0)
    assert(fy != 0)
    for j in range(joints.shape[0]):
        joints[j] = map(int, np.multiply(joints[j], [fx,fy]))
    return (resizedImage, joints)

def getNumChannelsLayer(net, layer_name):
    return net.blobs[layer_name].channels

def getBatchSizeLayer(net, layer_name):
    return net.blobs[layer_name].num

def netForward(net, imgch):
    """Run the model with the given input"""
    net.blobs['data'].data[...] = imgch
    net.forward() 

def getOutputLayer(net, layer_name):
    """Get output of layer_name layer"""
    if (net.blobs.get(layer_name).data.shape[0] == 1):
        return net.blobs.get(layer_name).data[0]
    return net.blobs.get(layer_name).data

def getDiffLayer(net, layer_name):
    """Get gradients of layer_name layer"""
    return net.blobs.get(layer_name).diff[0]

def getParamLayer(net, layer_name):
    return net.params[layer_name][0].data
    
def restoreSize(channels, channels_size, box_points):
    """Given the channel, it resize and place the channel in the right position
    in order to have a final estimation in the original image coordinate system.
    Channel has the format (c x w x h) """
    assert(channels.ndim == 3)
    num_channels = channels.shape[0]
    channels = channels.transpose(1,2,0)
    new_img = cv2.resize(channels, (box_points[2],box_points[3]), interpolation = cv2.INTER_LANCZOS4)

    reecreated_img = np.zeros((channels_size[0], channels_size[1], num_channels))
    reecreated_img[box_points[1]:box_points[1]+box_points[3],box_points[0]:box_points[0]+box_points[2]] = new_img
    return reecreated_img

def findPredictions(njoints, heatMaps):
    """Given the heat-maps returned from the NN, it returnes the joint positions.
    HeatMaps are in the format (w x h x c)"""
    assert(njoints == (heatMaps.shape[2]-1))
    predictions = np.empty((njoints, 2))
    for i in range(njoints):
        predictions[i] = findPoint(heatMaps[:,:,i])
    # +1 becuase we are considering the Matlab notation
    return (predictions + 1)

def getMasksFromFilename(file_name):
    """Given the name of a file from train or test set, it returns
    the masks about that specific frame"""
    while (file_name.find('/') >= 0):
        file_name = file_name[file_name.find('/')+1:]
    file_name = file_name[:file_name.find('.jpg')]
    data = file_name.split('_')
    
    fno = data[4]
    person = data[0]
    action = data[1]
    camera = data[3]
    return (fno, camera, action, person)

def generateMaskChannel(size, frame_num, camera, action, person):
    """Generate the metadata channel"""
    metadata = np.zeros((size, size))
    metadata[0,0] = frame_num
    metadata[0,1] = camera
    metadata[0,2] = action
    metadata[0,3] = person
    return metadata[:,:,np.newaxis]

def checkXY(joints):
    """Check if data are [x,y,x,y,...] and not [x,y,z,x,y,z,...]"""
    num_elems = len(np.array(joints).flatten())
    return ((np.mod(num_elems,2)==0) and (np.mod(num_elems,3)!=0)) 

def xyJoints(linearisedJoints):
    """Given a vector of joints it returns the joint positions in
    the [[x,y],[x,y]...] format"""
    num_elems = len(np.array(linearisedJoints).flatten())
    assert(np.mod(num_elems,2) == 0)
    if (type(linearisedJoints) == list):
        linearisedJoints = np.array(linearisedJoints)
    xy = linearisedJoints.reshape((num_elems/2, 2))
    return xy

def xyzJoints(linearisedJoints):
    """Given a vector of joints it returns the joint positions in
    the [[x,y,z],[x,y,z]...] format"""
    num_elems = len(np.array(linearisedJoints).flatten())
    assert(np.mod(num_elems,3) == 0)
    if (type(linearisedJoints) == list):
        linearisedJoints = np.array(linearisedJoints)
    xyz = linearisedJoints.reshape((num_elems/3, 3))
    return xyz

def checkJointsNonLinearised(joints):
    """Check if joints are in the [[x,y],[x,y],...] or [[x,y,z],[x,y,z],...] format"""
    try:
        check = np.shape(joints)[0] > np.shape(joints)[1]
        return check
    except:
        return False

def filterJoints(joints):
    """From the whole set of joints it removes those that are not used in 
    the error computation.
    Joints is in the format [[x,y],[x,y]...] or [[x,y,z],[x,y,z]...]"""
    if (type(joints) == list):
        joints = np.array(joints)
    if not checkJointsNonLinearised(joints):
        if checkXY:
            joints = xyJoints(joints)
        else:
            joints = xyzJoints(joints)
    joints_idx = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
    new_joints = joints[joints_idx]
    return np.array(new_joints)

def removeZ(joints):
    """From [x,y,z,x,y,z,...] set of joint positions remove the z elements, returning
    [x,y,x,y,...] list of joints."""
    joints = np.array(joints).flatten()
    joints = np.delete(joints, range(2, len(joints), 3))
    return xyJoints(joints)

def plotHeatMap(heatMap, secondHeatMaps=[], title=False):
    """Plot single heat-map"""
    if (len(secondHeatMaps) == 0):
        plt.imshow(heatMap)
    else:
        plt.subplot(121), plt.imshow(heatMap)
        plt.subplot(122), plt.imshow(secondHeatMaps)
    if title:
        if (len(secondHeatMaps) == 0):
            plt.title(title)
        else:
            plt.suptitle(title)
    plt.axis('off')

def plotHeatMaps(heatMaps, secondHeatMaps=[], title=False):
    """Plot heat-maps one after the other"""
    # check format
    if ((heatMaps.shape[0]==heatMaps.shape[1]) or (heatMaps.shape[-1]==18)):
        heatMaps = heatMaps.transpose(2,0,1)
    if (len(secondHeatMaps) != 0):
        if ((secondHeatMaps.shape[0]==secondHeatMaps.shape[1]) or (secondHeatMaps.shape[-1]==18)):
            secondHeatMaps = secondHeatMaps.transpose(2,0,1)
    # plot heat-maps
    for i in range(heatMaps.shape[0]):
            if (len(secondHeatMaps) != 0):
                plotHeatMap(heatMaps[i], secondHeatMaps[i], title)
            else:
                plotHeatMap(heatMaps[i], secondHeatMaps, title)
            plt.waitforbuttonpress()

def getConnections():
    conn = [[0, 1],[1, 2],[2, 3],[0, 4],[4, 5],[5, 6],[0, 7],[7, 8],[8, 9],
            [9, 10],[8, 11],[11, 12],[12, 13],[8, 14],[14, 15],[15, 16]]
    return conn

def plotImageJoints(image, joints, joints2=[], h=False):
    """Plot the image and the joint positions"""
    
    def getJointColor(j):
        colors = [(255,255,0),(255,0,255),(0,0,255),(0,255,255),(255,0,0),(0,255,0)]
        c = 0
        if j in range(1,4):
            c = 1
        if j in range(4,7):
            c = 2
        if j in range(9,11):
            c = 3
        if j in range(11,14):
            c = 4
        if j in range(14,17):
            c = 5
        return colors[c]
    
    img = convertImgCv2(image)
    marker_size = 4
    line_size = 2
    if not checkJointsNonLinearised(joints):
        joints = xyJoints(np.array(joints))
    if (len(joints2) != 0):
        joints2 = xyJoints(np.array(joints2).flatten())
        for j in range(joints.shape[0]):
            cv2.circle(img, (int(joints[j][0]), int(joints[j][1])), 3, (255, 0, 0), -1)
            if (len(joints2) != 0):
                cv2.circle(img, (int(joints2[j][0]), int(joints2[j][1])), 3, (0, 0, 255), -1)
    else:
        conn = getConnections()
        for c in conn:
            cv2.line(img, tuple(joints[c[0]].astype(int)),
                          tuple(joints[c[1]].astype(int)), getJointColor(c[0]), line_size, 16)
        for j in range(joints.shape[0]):
            cv2.circle(img, (int(joints[j][0]), int(joints[j][1])),
                       marker_size, getJointColor(j), -1)
    if not h:                  
        plt.imshow(img)
    return img
    
def plotJoints(joints, joints2=[], img=False):
    """Plot the image and the joint positions"""
    if not checkJointsNonLinearised(joints):
        joints = xyJoints(np.array(joints))
    if (len(joints2) != 0):
        joints2 = xyJoints(np.array(joints2).flatten())
    for j in range(joints.shape[0]):
        plt.scatter(joints[:,0],joints[:,1], color='r')
        if (len(joints2) != 0):
            plt.scatter(joints2[:,0],joints2[:,1], color='b')
    axes = plt.gca()
    axes.axis('equal')
    plt.show()

def plot3DJoints(joints, save_pdf=False):
    import mpl_toolkits.mplot3d.axes3d as p3
    
    def getJointColor(j):
        colors = [(0,0,0),(255,0,255),(0,0,255),(0,255,255),(255,0,0),(0,255,0)]
        c = 0
        if j in range(1,4):
            c = 1
        if j in range(4,7):
            c = 2
        if j in range(9,11):
            c = 3
        if j in range(11,14):
            c = 4
        if j in range(14,17):
            c = 5
        return colors[c]
    
    joints[2] -= joints[2].min()
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    conn = getConnections()
    for c in conn:
        col = '#%02x%02x%02x' % getJointColor(c[0])
        ax.plot([joints[0,c[0]], joints[0,c[1]]],
                [joints[1,c[0]], joints[1,c[1]]],
                [joints[2,c[0]], joints[2,c[1]]], c=col)
    for j in range(joints.shape[1]):
        col = '#%02x%02x%02x' % getJointColor(j)
        ax.scatter(joints[0,j], joints[1,j], joints[2,j], c=col, marker='o', edgecolor=col)
    p_v_x = 1#2
    p_v_y = 1#3
    diff_x = joints[0].max() - joints[0].min()
    diff_y = joints[1].max() - joints[1].min()
    diff_z = joints[2].max() - joints[2].min()
    prop_x = diff_x*p_v_x/diff_z
    prop_y = diff_y*p_v_y/diff_z
    ax.set_zlim3d(0, joints[2].max())
    ax = fig.gca(projection = '3d')
    ax.pbaspect = [prop_x, prop_y, 1]
    
    # Defining style
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    xticks_v = [joints[0].min()+0.1*diff_x, 0, joints[0].max()-0.1*diff_x]
    yticks_v = [joints[1].min()+0.1*diff_y, 0, joints[1].max()-0.1*diff_y]
    rot = 45
    plt.xticks(xticks_v, rotation=rot)
    plt.yticks(yticks_v, rotation=-rot)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.show()
    if save_pdf:
        with PdfPages(save_pdf) as pdf:
            pdf.savefig()
            plt.close()
    
def convertImgCv2(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
def plotCv2Image(img):
    img = convertImgCv2(img)
    plt.imshow(img)

def loadJsonFile(json_file):
    """Load json file"""
    with open(json_file) as data_file:
        data_this = json.load(data_file)
        data = np.array(data_this['root'])
        return (data, len(data))

def getActionNames():
    names = ['Directions', 'Discussion', 'Eating',
             'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
             'SittingDown', 'Smoking', 'TakingPhoto', 'Waiting', 'Walking',
             'WalkingDog', 'WalkingTogether']
    return [{'action':names[i],'idx':i+2} for i in range(len(names))]

def getCaffeCpm():
    """Get caffe com dir path"""
    return os.environ.get('CAFFE_HOME_CPM')+'/models/cpm_architecture'

def computeError(gt, pred):
    """Compute the euclidean distance between ground truth and predictions"""
    assert(pred.shape[0] > pred.shape[1])
    assert(gt.shape[0] == pred.shape[0])
    err = np.sqrt(np.power(gt-pred,2).sum(1)).mean()
    return err

def checkFileExists(file_path):
    try:
        return os.path.isfile(file_path)
    except:
        return False

def checkDirExists(dir_path):
    try:
        return os.path.isdir(dir_path)
    except:
        return False
    
