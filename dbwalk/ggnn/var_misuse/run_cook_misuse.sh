#!/bin/bash

data_root=$HOME/data/dataset/dbwalk/ggnn
data_name=gnn_varmisuse
lang=python

python -m dbwalk.ggnn.data_util.cook_ast_graphs \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -data_chunk_size 50000 \
    $@
