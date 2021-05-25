#!/bin/bash

data_root=$HOME/data/dataset/dbwalk/code2seq
data_name=varmisuse

bsize=32
nlayer=4
embed=256
hidden=512
max_steps=25
export CUDA_VISIBLE_DEVICES=1

save_dir=$HOME/scratch/results/dbwalk/code2seq/$data_name/b-$bsize-emb-$embed-nl-$nlayer-ms-$max_steps

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi


python -m dbwalk.code2seq.var_misuse.code2sec_var_misuse \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -data $data_name \
    -batch_size $bsize \
    -embed_dim $embed \
    -dim_feedforward $hidden \
    -transformer_layers $nlayer \
    -max_steps $max_steps \
    -iter_per_epoch 1000 \
    -learning_rate 1e-3 \
    -num_proc 0 \
    -gpu -1 \
    $@
