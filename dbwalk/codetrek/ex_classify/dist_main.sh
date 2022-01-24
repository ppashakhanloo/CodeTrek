#!/bin/bash

data_root=$HOME/CodeTrek_Material/dataset/codetrek
data_name=exception

bsize=2
nlayer=4
setenc=deepset

min_steps=16
max_steps=24
num_walks=100

num_proc=6
num_train_proc=4
gpu_list=0,1,2,3

export CUDA_VISIBLE_DEVICES=$gpu_list

save_dir=$HOME/CodeTrek_Material/results/codetrek/$data_name

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python -m dbwalk.codetrek.ex_classify.main \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -data $data_name \
    -set_encoder $setenc \
    -batch_size $bsize \
    -transformer_layers $nlayer \
    -num_proc $num_proc \
    -min_steps $min_steps \
    -max_steps $max_steps \
    -gpu -1 \
    -num_train_proc $num_train_proc \
    $@
