#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/asts
data_name=varshadow
lang=python

python -m dbwalk.code2seq.data_util.cook_ast_trees \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -use_node_val True \
    -data_chunk_size 50000 \
    $@
