#!/bin/bash

#GLOG_logtostderr=1 build/tools/caffe train \
#-solver models/cpm_architecture/prototxt/pose_solver.prototxt \
#-weights models/cpm_architecture/prototxt/caffemodel/trial_5/pose_iter_50000.caffemodel \
#-gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log.txt

# GLOG_logtostderr=1 build/tools/caffe train \
# -solver models/cpm_architecture/prototxt/pose_solver.prototxt \
# -weights models/cpm_architecture/prototxt/caffemodel/trial_5/pose_iter_50000.caffemodel \
# -gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log_merging_init.txt


# GLOG_logtostderr=1 build/tools/caffe train \
# -solver models/cpm_architecture/prototxt/pose_solver.prototxt \
# -weights models/cpm_architecture/prototxt/caffemodel/trial_5/pose_iter_70000.caffemodel \
# -gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log.txt

GLOG_logtostderr=1 build/tools/caffe train \
-solver models/cpm_architecture/prototxt/pose_solver.prototxt \
-snapshot models/cpm_architecture/prototxt/caffemodel/manifold_samearch3/pose_iter_90500.solverstate \
-gpu 0 2>&1 | tee models/cpm_architecture/prototxt/log2.txt