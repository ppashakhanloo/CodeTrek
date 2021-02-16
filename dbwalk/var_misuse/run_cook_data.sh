#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=walks_misuse
chunk_size=50000

python -m dbwalk.data_util.cook_data \
    -data_dir $data_root \
    -data $data_name \
    -data_chunk_size $chunk_size \
    $@
