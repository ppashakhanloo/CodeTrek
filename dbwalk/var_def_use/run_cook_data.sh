#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=walks
chunk_size=50000

python cook_data.py \
    -data_dir $data_root \
    -data $data_name \
    -data_chunk_size $chunk_size \
    $@
