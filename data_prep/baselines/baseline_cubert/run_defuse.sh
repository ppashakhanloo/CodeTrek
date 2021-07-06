#!/bin/bash

split_dir=$1 # splits
bucket=$2 # defuse
home_path=$3 # /home/pardisp/relational-representation
pred_kind=$4 # prog_cls, loc_cls
vocab=vocab.txt

export PYTHONPATH="$home_path"

rm -f $split_dir/*-done
rm -f $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python gen_samples.py $file_path $bucket tables cubert_"$pred_kind" $vocab defuse $pred_kind
  echo $f $?
done
