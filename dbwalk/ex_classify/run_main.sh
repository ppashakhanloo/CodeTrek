#!/bin/bash

data_root=$HOME/data/dataset/dbwalk
data_name=exception

bsize=32
embed=256
nlayer=4
nhead=8
hidden=512
dropout=0
setenc=deepset
online=True
num_proc=8
shuffle_var=False

export CUDA_VISIBLE_DEVICES=0

save_dir=$HOME/scratch/results/dbwalk/exception/gen-$online-b-$bsize-emb-$embed-nl-$nlayer-head-$nhead-hid-$hidden-dp-$dropout-set-$setenc-sv-$shuffle_var

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python main.py \
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
    -learning_rate 1e-4 \
    -min_steps 4 \
    -max_steps 20 \
    -gpu 0 \
    $@
