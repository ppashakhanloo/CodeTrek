#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/aug_asts
data_name=varmisuse

lv=5

export CUDA_VISIBLE_DEVICES=1

save_dir=$HOME/CodeTrek_Material/result/ggnn/varmisuse

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python -m dbwalk.ggnn.var_misuse.main \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -batch_size 2 \
    -max_lv $lv \
    -iter_per_epoch 10000 \
    -num_proc 0 \
    -gpu -1 \
    $@
