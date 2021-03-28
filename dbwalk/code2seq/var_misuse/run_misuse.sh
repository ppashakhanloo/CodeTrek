#!/bin/bash

data_root=$HOME/data/dataset/dbwalk/code2seq
data_name=debug_code2seq_misuse

bsize=11
nlayer=1
embed=16
hidden=512

export CUDA_VISIBLE_DEVICES=0

save_dir=$HOME/scratch/results/dbwalk/code2seq/$data_name/b-$bsize-emb-$embed-nl-$nlayer

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi


python code2sec_var_misuse.py \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -data $data_name \
    -batch_size $bsize \
    -embed_dim $embed \
    -dim_feedforward $hidden \
    -transformer_layers $nlayer \
    -iter_per_epoch 1000 \
    -learning_rate 1e-3 \
    -num_proc 0 \
    -gpu 0 \
    $@
