#!/bin/bash

data_root=$HOME/data/dataset/dbwalk/ggnn
data_name=debug_single

python cook_dict.py \
    -data_dir $data_root \
    -data $data_name \
    $@
