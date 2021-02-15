#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=tiny_except
lang=python
min_step=8
max_step=16
n_walk=100


python -m dbwalk.data_util.cook_from_gv \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -min_steps $min_step \
    -max_steps $max_step \
    -num_walks $n_walk \
    $@
