# Caffe (CPM Data Layer)
CPM architecture adapted in order to use the **Human3.6M** dataset's set of joints with the implementation of the manifold layer.
The architecture includes the manifold layer in stages: 4,5.

IN THIS VERSION RATHER THAN USING THE MANIFOLD LAYER JUST IN STAGE 4 AND 5 WE TRY TO USE IT ON ALL STAGES.

## Data dependencies
0. */data/Human3.6M/Data/* has to contain all the subject directories for testing and training the model
0. *models/cpm_architecture/data* contains the learned model for the manifold layer
0. *python/manifold/* contains all the dependencies for the manifold layer
0. *models/cpm_architecture/jsonDatasets* contains the files for training and testing
   * *H36M_annotations.json* (optional for testing)
   * *H36M_annotations_testSet.json*
   * *H36M_masks.mat*
0. *models/cpm_architecture/lmdb* contains the train lmdb database used for training the model (optional for testing)

## Python dependencies
<pre>
sudo pip install protobuf
sudo pip install scikit-image
sudo pip install matplotlib
sudo apt-get install python-tk
sudo pip install mpldatacursor
sudo apt-get install python-yaml
</pre>

## Fine-tuning a CNN for detection with Caffe
<pre>
  GLOG_logtostderr=1 build/tools/caffe train \
  -solver models/my_dir/caltech_finetune_solver_original.prototxt \
  -weights models/my_dir/bvlc_reference_caffenet.caffemodel \
  -gpu 0 2>&1 | tee results/log.txt
</pre>

## Multiple GPUs

Append after all the commands
<pre>
  --gpu=0,1
</pre>
for using two GPUs.

**NOTE**: each GPU runs the batchsize specified in your train_val.prototxt.  So if you go from 1 GPU to 2 GPU, your effective batchsize will double.  e.g. if your train_val.prototxt specified a batchsize of 256, if you run 2 GPUs your effective batch size is now 512.  So you need to adjust the batchsize when running multiple GPUs and/or adjust your solver params, specifically learning rate.

### Hardware Configuration Assumptions

The current implementation uses a tree reduction strategy.  e.g. if there are 4 GPUs in the system, 0:1, 2:3 will exchange gradients, then 0:2 (top of the tree) will exchange gradients, 0 will calculate
updated model, 0\-\>2, and then 0\-\>1, 2\-\>3. 

For best performance, P2P DMA access between devices is needed. Without P2P access, for example crossing PCIe root complex, data is copied through host and effective exchange bandwidth is greatly reduced.

Current implementation has a "soft" assumption that the devices being used are homogeneous.  In practice, any devices of the same general class should work together, but performance and total size is limited by the smallest device being used.  e.g. if you combine a TitanX and a GTX980, performance will be limited by the 980.  Mixing vastly different levels of boards, e.g. Kepler and Fermi, is not supported.

"nvidia-smi topo -m" will show you the connectivity matrix.  You can do P2P through PCIe bridges, but not across socket level links at this time, e.g. across CPU sockets on a multi-socket motherboard.

## CNN profiling

<pre>
  caffe time -model /path/to/file/structure.prototxt -iterations 10
</pre>
By default this is executed in CPU-mode. If instead a GPU-mode profiling is required, this is the command:
<pre>
  caffe time -model /path/to/file/structure.prototxt -gpu 0 -iterations 10
</pre>
