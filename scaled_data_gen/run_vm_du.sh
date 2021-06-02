#!/bin/bash

split_dir=$1 # splits_vm
src_bkt=$2 # varmisuse-defuse
home_path="$3" # /home/pardisp/relational-representation
walks_or_stubs="$4" # walks, stubs

export PYTHONPATH="$home_path"

rm $split_dir/*-done
rm $split_dir/*-log

for f in `ls -1 $split_dir`
do
  file_path=$split_dir/$f
  screen -S $f -d -m python vm_du_graphgen.py $file_path $src_bkt $src_bkt tables $walks_or_stubs $home_path $walks_or_stubs
  echo $f $?
done
