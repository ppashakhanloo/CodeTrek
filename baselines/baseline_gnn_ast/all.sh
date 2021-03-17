#!/bin/bash

PYTHON_CONVERTOR=2to3-2.7

function run_varmisuse {
  category=$1
  # ash11
  data_dir=/home/aadityanaik/relational-representation/datasets/cubert/py_files
  correct_dir=$data_dir/$category/correct
  misuse_dir=$data_dir/$category/misuse
  
  num=`ls -1 $data_dir/$category/correct | wc -l`
  mkdir -p graphs/varmisuse/$category

  for ((i=0;i<=$num;i++));
  do	
     corr=$((i*2))
     incorr=$((corr+1))
     printf $corr,$incorr,
     python3 gen_graph_jsons.py $correct_dir/file_$corr/source.py $misuse_dir/file_$incorr/source.py correct graphs/varmisuse/$category/file_$corr.json varmisuse
     python3 gen_graph_jsons.py $misuse_dir/file_$incorr/source.py $correct_dir/file_$corr/source.py misuse graphs/varmisuse/$category/file_$incorr.json varmisuse
done
}


function run_exception {
  # ash10
  data_dir=/data1/aadityanaik/allwalks

  mkdir -p graphs/exception/dev
  mkdir -p graphs/exception/eval
  mkdir -p graphs/exception/train

  for d in `ls -1 $data_dir` ;
  do
    printf $d,
    label=`cat $data_dir/$d/exception_info.txt`
    IFS='_' read -ra splits <<< "$d"
    category=${splits[2]}
    num=${splits[4]}
    cp $data_dir/$d/source.py . >& /dev/null
    $PYTHON_CONVERTOR -w source.py >& /dev/null
    python3 gen_graph_jsons.py source.py None $label graphs/exception/$category/graph_$num.json exception
  done
}


run_exception
# run_varmisuse dev
# run_varmisuse eval
# run_varmisuse train
