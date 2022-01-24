#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/aug_asts
data_name=defuse
lang=python

python -m dbwalk.ggnn.data_util.cook_ast_graphs \
    -data_dir $data_root \
    -data $data_name \
    -use_node_val True \
    -language $lang \
    -data_chunk_size 50000 \
    $@
