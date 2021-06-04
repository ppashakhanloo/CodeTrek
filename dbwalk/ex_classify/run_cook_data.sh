#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=ex_large_walks
chunk_size=1000
use_node_val=True

python -m dbwalk.data_util.cook_data \
    -data_dir $data_root \
    -data $data_name \
    -data_chunk_size $chunk_size \
    -use_node_val $use_node_val \
    $@
