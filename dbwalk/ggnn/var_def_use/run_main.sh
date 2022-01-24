#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/aug_asts
data_name=defuse

lv=5

export CUDA_VISIBLE_DEVICES=2

save_dir=$HOME/CodeTrek_Material/result/ggnn/exception

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python -m dbwalk.ggnn.var_def_use.main \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -batch_size 2 \
    -max_lv $lv \
    -iter_per_epoch 300 \
    -learning_rate 1e-4 \
    -num_proc 0 \
    -gpu -1 \
    $@
