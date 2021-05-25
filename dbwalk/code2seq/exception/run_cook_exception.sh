#!/bin/bash

data_root=$HOME/data/dataset/dbwalk/code2seq
data_name=exception
lang=python

python -m dbwalk.code2seq.data_util.cook_ast_trees \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -data_chunk_size 50000 \
    $@
