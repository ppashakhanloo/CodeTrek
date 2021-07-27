#!/bin/bash

split_dir=$1 # splits
bucket=$2
pred_kind=$3
task=$4

rm -f $split_dir/*-done
rm -f $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python build_ast_for_codetrek.py $bucket tables cq_ast_$pred_kind $split_dir/$f $task $pred_kind
  echo $f $?
done
