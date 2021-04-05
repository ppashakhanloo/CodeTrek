#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=defuse
lang=python

python -m dbwalk.data_util.cook_from_gv_stub \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -data_chunk_size 5000 \
    $@
