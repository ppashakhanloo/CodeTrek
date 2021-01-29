#!/bin/bash

data_dir=/home/aadityanaik/relational-representation/datasets/cubert/py_files

function run { #category
  category=$1
  correct_dir=$data_dir/$category/correct
  misuse_dir=$data_dir/$category/misuse
  
  num=`ls -1 $data_dir/$category/correct | wc -l`
  mkdir -p graphs/$category

  for ((i=0;i<$num;i++));
  do	
     corr=$((i*2))
     incorr=$((corr+1))
     printf $corr $incorr
     python3 gen_graph_jsons.py $correct_dir/file_$corr/source.py $misuse_dir/file_$incorr/source.py correct graphs/$category/file_$corr.json
     python3 gen_graph_jsons.py $misuse_dir/file_$incorr/source.py $correct_dir/file_$corr/source.py misuse graphs/$category/file_$incorr.json
  done
}

run dev
run eval
run train
