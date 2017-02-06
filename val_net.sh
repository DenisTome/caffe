#!/bin/bash

python models/cpm_architecture/runValidation.py \
models/cpm_architecture/prototxt/caffemodel/prob_model/to_evaluate \
-s 250 -o eval_stage6
