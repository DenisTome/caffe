#!/bin/bash

# python models/cpm_architecture/runCaffeTest.py \
# models/cpm_architecture/prototxt/pose_deploy_stg4.prototxt \
# models/cpm_architecture/prototxt/caffemodel/manifold_samearch3/pose_iter_110000.caffemodel \
# --num_parts 5 \
# --run_part 1
# python models/cpm_architecture/runCaffeTest.py \
# models/cpm_architecture/prototxt/pose_deploy_stg4.prototxt \
# models/cpm_architecture/prototxt/caffemodel/manifold_samearch3/pose_iter_110000.caffemodel \
# --num_parts 5 \
# --run_part 2
# python models/cpm_architecture/runCaffeTest.py \
# models/cpm_architecture/prototxt/pose_deploy_stg4.prototxt \
# models/cpm_architecture/prototxt/caffemodel/manifold_samearch3/pose_iter_110000.caffemodel \
# --num_parts 5 \
# --run_part 3
# python models/cpm_architecture/runCaffeTest.py \
# models/cpm_architecture/prototxt/pose_deploy_stg4.prototxt \
# models/cpm_architecture/prototxt/caffemodel/manifold_samearch3/pose_iter_110000.caffemodel \
# --num_parts 5 \
# --run_part 4
# python models/cpm_architecture/runCaffeTest.py \
# models/cpm_architecture/prototxt/pose_deploy_stg4.prototxt \
# models/cpm_architecture/prototxt/caffemodel/manifold_samearch3/pose_iter_110000.caffemodel \
# --num_parts 5 \
# --run_part 5


# python models/cpm_architecture/runCaffeTest.py \
# models/cpm_architecture/prototxt/caffemodel/manifold_diffarch3/to_test_tmp/pose_deploy_stg4.prototxt \
# models/cpm_architecture/prototxt/caffemodel/manifold_diffarch3/to_test_tmp/pose_iter_22000.caffemodel \
# -o models/cpm_architecture/prototxt/caffemodel/manifold_diffarch3/to_test_tmp/ \
# --num_parts 5 \
# --run_part 1

#python models/cpm_architecture/runCaffeTest.py \
#models/cpm_architecture/prototxt/caffemodel/manifold_diffarch3/to_test_tmp/pose_deploy_stg4.prototxt \
#models/cpm_architecture/prototxt/caffemodel/manifold_diffarch3/to_test_tmp/pose_iter_22000.caffemodel \
#--merge_parts_dir /home/denitome/Desktop/tmp/ \
#--num_parts 5 \


#python models/cpm_architecture/runCaffeTest.py \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_deploy.prototxt \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_iter_10000.caffemodel \
#-o models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/ \
#--num_parts 5 \
#--run_part 1
#python models/cpm_architecture/runCaffeTest.py \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_deploy.prototxt \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_iter_10000.caffemodel \
#-o models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/ \
#--num_parts 5 \
#--run_part 2
#python models/cpm_architecture/runCaffeTest.py \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_deploy.prototxt \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_iter_10000.caffemodel \
#-o models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/ \
#--num_parts 5 \
#--run_part 3
#python models/cpm_architecture/runCaffeTest.py \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_deploy.prototxt \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_iter_10000.caffemodel \
#-o models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/ \
#--num_parts 5 \
#--run_part 4
#python models/cpm_architecture/runCaffeTest.py \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_deploy.prototxt \
#models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_iter_10000.caffemodel \
#-o models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/ \
#--num_parts 5 \
#--run_part 5

python models/cpm_architecture/runCaffeTest.py \
models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_deploy.prototxt \
models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/pose_iter_10000.caffemodel \
--merge_parts_dir models/cpm_architecture/prototxt/caffemodel/prob_model/to_test/ \
--num_parts 5 \
