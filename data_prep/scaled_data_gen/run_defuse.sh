#!/bin/bash

split_dir=$1 # du_splits
bucket=$2 # varmisuse-defuse
home_path="$3" # /home/pardisp/relational-representation
walks_or_graphs="$4" # walks, graphs
pred_kind="$5" # prog_cls, loc_cls

export PYTHONPATH="$home_path"

rm -f $split_dir/*-done
rm -f $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python defuse_graphgen.py $file_path $bucket tables "$walks_or_graphs"_"$pred_kind" $home_path $walks_or_graphs $pred_kind
  echo $f $?
done
