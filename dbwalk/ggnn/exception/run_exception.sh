#!/bin/bash

data_root=$HOME/data/dataset/dbwalk/ggnn
data_name=exception

lv=5

export CUDA_VISIBLE_DEVICES=1

save_dir=$HOME/scratch/results/dbwalk/ggnn/$data_name/lv-$lv

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python -m dbwalk.ggnn.exception.gnn_exception \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -batch_size 64 \
    -max_lv $lv \
    -iter_per_epoch 300 \
    -num_proc 0 \
    -gpu -1 \
    -learning_rate 1e-4 \
    $@
