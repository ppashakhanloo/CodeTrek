#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/codetrek
data_name=varshadow
lang=python

python -m dbwalk.data_util.cook_from_gv_stub \
    -data_dir $data_root \
    -data $data_name \
    -language $lang \
    -data_chunk_size 50000 \
    $@
