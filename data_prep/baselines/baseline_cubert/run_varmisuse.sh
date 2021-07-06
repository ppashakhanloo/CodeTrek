#!/bin/bash

split_dir=$1 # splits
bucket=$2 # varmisuse
home_path=$3 # /home/pardisp/relational-representation
vocab=vocab.txt

export PYTHONPATH="$home_path"

rm -f $split_dir/*-done
rm -f $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python gen_samples.py $file_path $bucket tables cubert_loc_cls $vocab varmisuse loc_cls
  echo $f $?
done
