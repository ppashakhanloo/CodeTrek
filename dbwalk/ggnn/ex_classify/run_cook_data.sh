#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/aug_asts
data_name=exception
lang=python

python -m dbwalk.ggnn.data_util.cook_ast_graphs \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -use_node_val True \
    -data_chunk_size 50000 \
    $@
