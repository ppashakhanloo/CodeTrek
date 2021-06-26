#!/bin/bash

split_dir=$1 # ex_splits
bucket=$2 # exception-small
home_path=$3 # /home/pardisp/relational-representation

export PYTHONPATH="$home_path"

rm -f $split_dir/*-done
rm -f $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python gen_graph_jsons.py $file_path $bucket tables ast_graphs exception none
  echo $f $?
done
