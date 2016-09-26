import os
import math
import caffe
from caffe import layers as L  # pseudo module using __getattr__ magic to generate protobuf messages
from caffe import params as P  # pseudo module using __getattr__ magic to generate protobuf messages

def setLayers(data_source, batch_size, layername, kernel, stride, outCH, label_name, transform_param_in, deploy=False):
    # it is tricky to produce the deploy prototxt file, as the data input is not from a layer, so we have to creat a workaround
    # producing training and testing prototxt files is pretty straight forward
    n = caffe.NetSpec()
    assert len(layername) == len(kernel)
    assert len(layername) == len(stride)
    assert len(layername) == len(outCH)

    # produce data definition for deploy net
    if deploy == False:
        # here we will return the new structure for loading h36m dataset
        n.data, n.tops['label'] = L.CPMData(cpmdata_param=dict(backend=1, source=data_source, batch_size=batch_size), 
                                                    transform_param=transform_param_in, ntop=2)
        n.tops[label_name[1]], n.tops[label_name[0]], n.tops[label_name[2]] = L.Slice(n.label, slice_param=dict(axis=1, slice_point=[18,36]), ntop=3)
        n.image, n.center_map = L.Slice(n.data, slice_param=dict(axis=1, slice_point=3), ntop=2)
    else:
        input = "data"
        dim1 = 1
        dim2 = 5
        dim3 = 368
        dim4 = 368
        # make an empty "data" layer so the next layer accepting input will be able to take the correct blob name "data",
        # we will later have to remove this layer from the serialization string, since this is just a placeholder
        n.data = L.Layer()
        # Slice layer slices input layer to multiple output along a given dimension
        # axis: 1 define in which dimension to slice
        # slice_point: 3 define the index in the selected dimension (the number of
        # indices must be equal to the number of top blobs minus one)
        # Considering input Nx3x1x1, by slice_point = 2
        # top1 : Nx2x1x1
        # top2 : Nx1x1x1
        n.image, n.center_map, n.tops[label_name[2]] = L.Slice(n.data, slice_param=dict(axis=1, slice_point=[3,4]), ntop=3)

    
    n.pool_center_lower = L.Pooling(n.center_map, kernel_size=9, stride=8, pool=P.Pooling.AVE)

    # just follow arrays..CPCPCPCPCCCC....
    last_layer = 'image'
    stage = 1
    conv_counter = 1
    last_manifold = 'NONE'
    pool_counter = 1
    drop_counter = 1
    state = 'image' # can be image or fuse
    share_point = 0
    
    for l in range(0, len(layername)):
        if layername[l] == 'C':
            if state == 'image':
                conv_name = 'conv%d_stage%d' % (conv_counter, stage)
            else:
                conv_name = 'Mconv%d_stage%d' % (conv_counter, stage)
            #if stage == 1:
            #    lr_m = 5
            #else:
            #    lr_m = 1
            if ((stage == 1 and conv_counter == 7) or
                (stage > 1 and state != 'image' and (conv_counter in [1, 5]))):
                conv_name = '%s_new' % conv_name
                #lr_m = 1e-3# 1 # changed -> using original model
            #else:
            lr_m = 1e-4 # 1e-3 (best res so far)
            
            # additional for python layer
            if (stage > 1 and state != 'image' and (conv_counter == 1)):
                conv_name = '%s_mf' % conv_name
                lr_m = 1
            n.tops[conv_name] = L.Convolution(n.tops[last_layer], kernel_size=kernel[l],
                                                  num_output=outCH[l], pad=int(math.floor(kernel[l]/2)),
                                                  param=[dict(lr_mult=lr_m, decay_mult=1), dict(lr_mult=lr_m*2, decay_mult=0)],
                                                  weight_filler=dict(type='gaussian', std=0.01),
                                                  bias_filler=dict(type='constant'))
            last_layer = conv_name
            if not(layername[l+1] == 'L' or layername[l+1] == 'M'):
                if(state == 'image'):
                    ReLUname = 'relu%d_stage%d' % (conv_counter, stage)
                    n.tops[ReLUname] = L.ReLU(n.tops[last_layer], in_place=True)
                else:
                    ReLUname = 'Mrelu%d_stage%d' % (conv_counter, stage)
                    n.tops[ReLUname] = L.ReLU(n.tops[last_layer], in_place=True)
                last_layer = ReLUname
            conv_counter += 1
        elif layername[l] == 'P': # Pooling
            n.tops['pool%d_stage%d' % (pool_counter, stage)] = L.Pooling(n.tops[last_layer], kernel_size=kernel[l], stride=stride[l], pool=P.Pooling.MAX)
            last_layer = 'pool%d_stage%d' % (pool_counter, stage)
            pool_counter += 1
        elif layername[l] == 'M':
            last_manifold = 'manifolds_stage%d' % stage
            debug_mode = 0
            parameters = '{"njoints": 17,"sigma": 1, "debug_mode": %r, "max_area": 100, "percentage_max": 3, "train": %u, "Lambda": %.3f }' % (debug_mode, not deploy, 0.05)
            n.tops[last_manifold] = L.Python(n.tops[last_layer],n.tops[label_name[2]],python_param=dict(module='newheatmaps',layer='MyCustomLayer',param_str=parameters))#,loss_weight=1)
        elif layername[l] == 'L':
            # Loss: n.loss layer is only in training and testing nets, but not in deploy net.
            if deploy == False:
                if stage == 1:
                    n.tops['loss_stage%d' % stage] = L.EuclideanLoss(n.tops[last_layer], n.tops[label_name[0]])
                else:
                    n.tops['loss_stage%d' % stage] = L.EuclideanLoss(n.tops[last_layer], n.tops[label_name[1]])

            stage += 1
            last_connect = last_layer
            last_layer = 'image'
            conv_counter = 1
            pool_counter = 1
            drop_counter = 1
            state = 'image'
        elif layername[l] == 'D':
            if deploy == False:
                n.tops['drop%d_stage%d' % (drop_counter, stage)] = L.Dropout(n.tops[last_layer], in_place=True, dropout_param=dict(dropout_ratio=0.5))
                drop_counter += 1
        elif layername[l] == '@':
            n.tops['concat_stage%d' % stage] = L.Concat(n.tops[last_layer], n.tops[last_connect], n.tops[last_manifold], n.pool_center_lower, concat_param=dict(axis=1))
            #n.tops['concat_stage%d' % stage] = L.Concat(n.tops[last_layer], n.tops[last_connect], n.tops[last_manifold], n.pool_center_lower, concat_param=dict(axis=1))
                        
            #n.tops['concat_stage%d' % stage] = L.Concat(n.tops[last_layer], n.tops[last_connect], n.pool_center_lower, concat_param=dict(axis=1))
            conv_counter = 1
            state = 'fuse'
            last_layer = 'concat_stage%d' % stage
        elif layername[l] == '$':
            if not share_point:
                share_point = last_layer
            else:
                last_layer = share_point
    # final process
    stage -= 1
    if stage == 1:
        n.silence = L.Silence(n.pool_center_lower, ntop=0)

    if deploy == False:
        return str(n.to_proto())
        # for generating the deploy net
    else:
        # generate the input information header string
        deploy_str = 'input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"' + input + '"',
                                                                                                    dim1, dim2, dim3, dim4)
        # assemble the input header with the net layers string.  remove the first placeholder layer from the net string.
        return deploy_str + '\n' + 'layer {' + 'layer {'.join(str(n.to_proto()).split('layer {')[2:])



def writePrototxts(dataFolder, dir, batch_size, layername, kernel, stride, outCH, transform_param_in, folder_name, label_name, solver_param):
    # write the net prototxt files out
    with open('%s/pose_train_test.prototxt' % dir, 'w') as f:
        print 'writing %s/pose_train_test.prototxt' % dir
        str_to_write = setLayers(dataFolder, batch_size, layername, kernel, stride, outCH, label_name, transform_param_in, deploy=False)
        f.write(str_to_write)

    with open('%s/pose_deploy.prototxt' % dir, 'w') as f:
        print 'writing %s/pose_deploy.prototxt' % dir
        str_to_write = str(setLayers('', 0, layername, kernel, stride, outCH, label_name, transform_param_in, deploy=True))
        f.write(str_to_write)

    with open('%s/pose_solver.prototxt' % dir, "w") as f:
        solver_string = getSolverPrototxt(path_in_caffe, folder_name, dir, solver_param)
        print 'writing %s/pose_solver.prototxt' % dir
        f.write('%s' % solver_string)


def getSolverPrototxt(path_in_caffe, folder_name, dir, solver_param):
    maxiter = (int)(solver_param['train_size']/solver_param['batch_size']*solver_param['num_epochs'])
    test_iter = (int)(solver_param['test_size']/solver_param['batch_size']*solver_param['num_epochs'])
    
    string = 'net: "%s/%s/pose_train_test.prototxt"\n' % (path_in_caffe, dir)
    string += '#test_iter: %d\n#test_intervall: %d\n' % (test_iter, solver_param['test_interval'])
    string += 'base_lr: %f\n' % solver_param['base_lr']
    string += 'momentum: 0.9\n'
    string += 'weight_decay: %f\n' % solver_param['weight_decay']
    # The learning rate policy
    if not solver_param['lr_policy_fixed']:
        string += 'lr_policy: "step"\n'
        string += 'stepsize: %d\n' % solver_param['stepsize']
    else:
        string += 'lr_policy: "fixed"\n'
    string += 'gamma: 0.333\n'
    # Display every n iterations
    string += 'display: %d\n' % solver_param['disp_iter']
    # The maximum number of iterations
    string += 'max_iter: %d\n' % maxiter
    # snapshot intermediate results
    string += 'snapshot: %d\n' % solver_param['snapshot']
    string += 'snapshot_prefix: "%s/%s/pose"\n' % (path_in_caffe, folder_name)
    # solver mode: CPU or GPU
    if solver_param['gpu']:
        string += 'solver_mode: GPU\n'
    else:
        string += 'solver_mode: CPU\n'
    return string

if __name__ == "__main__":

    ### SOLVER SETTINGS
    path_in_caffe = 'models/cpm_architecture'
    directory = 'prototxt'
    dataFolder = '%s/lmdb/train' % (path_in_caffe)
    batch_size = 1
    snapshot=100 #5000
    # base_lr = 1e-5 (8e-5)
    base_lr = 1e-4 #8e-5
    solver_param = dict(stepsize=50000, batch_size=batch_size, num_epochs=12, base_lr = base_lr,
                        train_size=115327, test_size=40649, test_interval=5000,
                        weight_decay=0.0005, lr_policy_fixed=False, disp_iter=5,
                        snapshot=snapshot, gpu=False)
    ### END

    d_caffemodel = '%s/caffemodel/manifold_merging2' % directory # the place you want to store your caffemodel

    # num_parts and np_in_lmdb are two parameters that are used inside the framework to move from one
    # dataset definition to another. Num_parts is the number of parts we want to have, while
    # np_in_lmdb is the number of joints saved in lmdb format using the dataset whole set of joints.
    transform_param = dict(stride=8, crop_size_x=368, crop_size_y=368, visualize=False,
                             target_dist=1, scale_prob=0.7, scale_min=0.7, scale_max=1.2,
                             max_rotate_degree=10, center_perterb_max=0, do_clahe=False, 
                             num_parts=17, np_in_lmdb=17, transform_body_joint=False)
    nCP = 3
    CH = 128
    stage = 6
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(d_caffemodel):
        os.makedirs(d_caffemodel)
    
    layername = ['C', 'P'] * nCP + ['C','C','D','C','D','C'] + ['M'] + ['L'] # first-stage
    kernel =    [ 9 ,  3 ] * nCP + [ 5 , 9 , 0 , 1 , 0 , 1 ] + [ 0 ] + [ 0 ] # first-stage
    outCH =     [128, 128] * nCP + [ 32,512, 0 ,512, 0 ,18 ] + [ 0 ] + [ 0 ] # first-stage
    stride =    [ 1 ,  2 ] * nCP + [ 1 , 1 , 0 , 1 , 0 , 1 ] + [ 0 ] + [ 0 ] # first-stage

    if stage >= 2:
        layername += ['C', 'P'] * nCP + ['$'] + ['C'] + ['@'] + ['C'] * 5            + ['M'] + ['L']
        outCH +=     [128, 128] * nCP + [ 0 ] + [32 ] + [ 0 ] + [128,128,128,128,18] + [ 0 ] + [ 0 ]
        kernel +=    [ 9,   3 ] * nCP + [ 0 ] + [ 5 ] + [ 0 ] + [11, 11, 11, 1,   1] + [ 0 ] + [ 0 ]
        stride +=    [ 1 ,  2 ] * nCP + [ 0 ] + [ 1 ] + [ 0 ] + [ 1 ] * 5            + [ 0 ] + [ 0 ]

        for s in range(3, stage+1):
            if (s != stage):
                layername += ['$'] + ['C'] + ['@'] + ['C'] * 5            + ['M'] + ['L']
                outCH +=     [ 0 ] + [32 ] + [ 0 ] + [128,128,128,128,18] + [ 0 ] + [ 0 ]
                kernel +=    [ 0 ] + [ 5 ] + [ 0 ] + [11, 11, 11,  1, 1 ] + [ 0 ] + [ 0 ]
                stride +=    [ 0 ] + [ 1 ] + [ 0 ] + [ 1 ] * 5            + [ 0 ] + [ 0 ]
            else:
                layername += ['$'] + ['C'] + ['@'] + ['C'] * 5            + ['L']
                outCH +=     [ 0 ] + [32 ] + [ 0 ] + [128,128,128,128,18] + [ 0 ]
                kernel +=    [ 0 ] + [ 5 ] + [ 0 ] + [11, 11, 11,  1, 1 ] + [ 0 ]
                stride +=    [ 0 ] + [ 1 ] + [ 0 ] + [ 1 ] * 5            + [ 0 ]

    label_name = ['label_1st_lower', 'label_lower', 'metadata']
    writePrototxts(dataFolder, directory, batch_size, layername, kernel, stride, outCH, transform_param, d_caffemodel, label_name, solver_param)