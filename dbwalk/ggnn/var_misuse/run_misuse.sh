#!/bin/bash

data_root=$HOME/data/dataset/dbwalk/ggnn
data_name=debug_ggnn_misuse

python var_misuse.py \
    -data_dir $data_root/$data_name \
    -batch_size 11 \
    -max_lv 3 \
    -num_proc 0 \
    $@
