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

function run_exception_large {
  # ash12
  category=$1
  data_dir=/data1/pardisp/exception_large_programs

  mkdir -p graphs/exception_large/$category

  for d in `ls -1 $data_dir/$category | grep -E "txt$"` ;
  do
    printf $d,
    label=`cat $data_dir/$category/$d`
    IFS='_' read -ra splits <<< "$d" >& /dev/null
    name=${splits[0]}
    num=${splits[1]}
    python_file=$name'_'$num'.py'
    $PYTHON_CONVERTOR -w $data_dir/$category/$python_file >& /dev/null
    python3 gen_graph_jsons.py $data_dir/$category/$python_file None $label graphs/exception_large/$category/graph_$python_file.json exception
  done
}

function run_defuse {
  # ash09
  category=$1
  data_dir=/data1/pardisp/defuse_files/$category

  mkdir -p graphs/defuse/$category

  for d in `ls -1 $data_dir | grep -E "\.txt$"` ;
  do
    printf $d,
    label=`cat $data_dir/$d`
    IFS='_' read -ra splits <<< "$d" >& /dev/null
    num=${splits[1]}
    IFS='.' read -ra splits <<< "$num" >& /dev/null
    num=${splits[0]}
    python_file="file_"$num.py
    $PYTHON_CONVERTOR -w $data_dir/$python_file >& /dev/null
    python3 gen_graph_jsons.py $data_dir/$python_file None $label graphs/defuse/$category/graph_$python_file.json defuse
  done

}


## UNCOMMENT ANY OF THE FUNCTION BELOW TO GENERATE AST-BASED GRAPHS.

#run_defuse dev
#run_defuse eval
#run_defuse train

#run_exception

#run_exception_large dev
#run_exception_large eval
#run_exception_large train

#run_varmisuse dev
#run_varmisuse eval
#run_varmisuse train
