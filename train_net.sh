#!/bin/bash

#GLOG_logtostderr=1 build/tools/caffe train \
#-solver models/cpm_architecture/prototxt/pose_solver.prototxt \
#-weights models/cpm_architecture/savedmodels/pose_iter_985000_addLEEDS.caffemodel \
#-gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log.txt

#GLOG_logtostderr=1 build/tools/caffe train \
#-solver models/cpm_architecture/prototxt/pose_solver.prototxt \
#-weights models/cpm_architecture/prototxt/caffemodel/trial_5/pose_iter_50000.caffemodel \
#-gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log.txt

GLOG_logtostderr=1 build/tools/caffe train \
-solver models/cpm_architecture/prototxt/pose_solver.prototxt \
-weights models/cpm_architecture/prototxt/caffemodel/trial_5/pose_iter_50000.caffemodel \
-gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log_merging.txt

# GLOG_logtostderr=1 build/tools/caffe train \
# -solver models/cpm_architecture/prototxt/pose_solver.prototxt \
# -weights models/cpm_architecture/prototxt/caffemodel/manifold_gt_input/pose_iter_1.caffemodel \
# -gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log_merging.txt


# GLOG_logtostderr=1 build/tools/caffe train \
# -solver models/cpm_architecture/prototxt/pose_solver.prototxt \
# -snapshot models/cpm_architecture/prototxt/caffemodel/pose_iter_7000.solverstate \
# -gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log_merging.txt

#GLOG_logtostderr=1 build/tools/caffe train \
#-solver models/cpm_architecture/prototxt/pose_solver.prototxt \
#-snapshot models/cpm_architecture/prototxt/caffemodel/pose_iter_50000.solverstate \
#-gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log.txt
