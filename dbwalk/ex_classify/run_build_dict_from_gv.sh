#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=exception_small
lang=python
min_step=4
max_step=20
n_walk=300


python3 -m dbwalk.data_util.cook_from_gv \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -min_steps $min_step \
    -max_steps $max_step \
    -num_walks $n_walk \
    $@
