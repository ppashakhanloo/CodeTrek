#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=tiny_except
lang=python

python -m dbwalk.data_util.cook_from_gv \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -data_chunk_size 50000 \
    $@
