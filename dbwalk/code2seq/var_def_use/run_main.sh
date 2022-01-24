#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/asts
data_name=defuse

bsize=16
nlayer=4
embed=256
hidden=512
max_steps=25

export CUDA_VISIBLE_DEVICES=0

save_dir=$HOME/CodeTrek_Material/results/code2seq/defuse

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi


python -m dbwalk.code2seq.var_def_use.main \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -data $data_name \
    -batch_size $bsize \
    -embed_dim $embed \
    -max_steps $max_steps \
    -dim_feedforward $hidden \
    -transformer_layers $nlayer \
    -iter_per_epoch 100 \
    -learning_rate 1e-3 \
    -num_proc 0 \
    -gpu -1 \
    $@
