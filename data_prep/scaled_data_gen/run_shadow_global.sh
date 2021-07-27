#!/bin/bash

split_dir=$1 # vm_splits
home_path="$2" # /home/pardisp/relational-representation
walks_or_graphs="$3" # walks, graphs

export PYTHONPATH="$home_path"

rm -f $split_dir/*-done
rm -f $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python shadow_global_graphgen.py $file_path $walks_or_graphs
  echo $f $?
done
