#!/bin/bash

split_dir=$1
src_bkt=$2
home_path="$3" # /home/pardisp/relational-representation

export PYTHONPATH="$home_path"

rm $split_dir/*-done
rm $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python3 exception_graphgen.py $file_path $src_bkt tables graphs $home_path
  echo $f $?
done
