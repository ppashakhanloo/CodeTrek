#!/bin/bash

split_dir=$1 # ex_splits
bucket=$2 # exception-small
home_path="$3" # /home/pardisp/relational-representation
walks_or_graphs="$4" # walks, graphs

export PYTHONPATH="$home_path"

rm -f $split_dir/*-done
rm -f $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python exception_graphgen.py $file_path $bucket tables $walks_or_graphs $home_path $walks_or_graphs
  echo $f $?
done
