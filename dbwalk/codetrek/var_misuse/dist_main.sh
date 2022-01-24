#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/codetrek
data_name=varmisuse

bsize=2
nlayer=4
setenc=deepset

min_steps=4
max_steps=16
num_walks=100

num_proc=6
gpu_list=0,1,2,3
num_train_proc=4

export CUDA_VISIBLE_DEVICES=$gpu_list

save_dir=$HOME/CodeTrek_Material/results/codetrek/$data_name


if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python -m dbwalk.codetrek.var_misuse.main \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -data $data_name \
    -set_encoder $setenc \
    -batch_size $bsize \
    -transformer_layers $nlayer \
    -num_proc $num_proc \
    -min_steps $min_steps \
    -max_steps $max_steps \
    -num_walks $num_walks \
    -gpu -1 \
    $@
