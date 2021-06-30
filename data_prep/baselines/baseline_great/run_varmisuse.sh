#!/bin/bash

split_dir=$1 # vm_splits
bucket=$2 # varmisuse
home_path=$3 # /home/pardisp/relational-representation
pred_kind=$4 # prog_cls, loc_cls, loc_rep

export PYTHONPATH="$home_path"

rm -f $split_dir/*-done
rm -f $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python gen_graph_jsons.py $file_path $bucket tables great_"$pred_kind" varmisuse $pred_kind
  echo $f $?
done
