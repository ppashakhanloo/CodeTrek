#!/bin/bash

data_root=$HOME/data/dataset/dbwalk/ggnn
data_name=debug_single

python var_misuse.py \
    -data_dir $data_root/$data_name \
    -batch_size 1 \
    -num_proc 0 \
    $@
