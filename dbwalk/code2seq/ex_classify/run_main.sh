#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/asts
data_name=exception

bsize=64
nlayer=4
embed=256
hidden=512
max_steps=25
export CUDA_VISIBLE_DEVICES=1

save_dir=$HOME/CodeTrek_Material/result/code2seq/exception

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi


python -m dbwalk.code2seq.ex_classify.main \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -data $data_name \
    -batch_size $bsize \
    -embed_dim $embed \
    -dim_feedforward $hidden \
    -transformer_layers $nlayer \
    -max_steps $max_steps \
    -iter_per_epoch 600 \
    -learning_rate 1e-3 \
    -num_proc 0 \
    -gpu -1 \
    $@
