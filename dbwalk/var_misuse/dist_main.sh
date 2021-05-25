#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=varmisuse

bsize=64
embed=256
nlayer=4
nhead=8
hidden=512
dropout=0
setenc=deepset
online=True
num_proc=6
shuffle_var=False
use_node_val=True

export CUDA_VISIBLE_DEVICES=0,1,2,3

save_dir=$HOME/scratch/results/dbwalk/$data_name/gen-$online-b-$bsize-emb-$embed-nl-$nlayer-head-$nhead-hid-$hidden-dp-$dropout-set-$setenc-sv-$shuffle_var-nv-$use_node_val

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python -m dbwalk.var_misuse.main \
    -data_dir $data_root/$data_name \
    -save_dir $save_dir \
    -data $data_name \
    -online_walk_gen $online \
    -set_encoder $setenc \
    -shuffle_var $shuffle_var \
    -batch_size $bsize \
    -embed_dim $embed \
    -nhead $nhead \
    -transformer_layers $nlayer \
    -dim_feedforward $hidden \
    -dropout $dropout \
    -iter_per_epoch 1000 \
    -num_proc $num_proc \
    -use_node_val $use_node_val \
    -learning_rate 1e-4 \
    -min_steps 16 \
    -max_steps 24 \
    -gpu_list 0,1,2,3 \
    -port 29500 \
    -num_train_proc 4 \
    $@
