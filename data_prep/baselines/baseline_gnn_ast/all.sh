#!/bin/bash

PYTHON_CONVERTOR=2to3-2.7

function run_varmisuse {
  category=$1
  indir=$2
  outdir=$3
  pred_kind=$4

  correct_dir=$indir/$category/correct
  misuse_dir=$indir/$category/misuse

  num=`ls -1 $indir/$category/correct | wc -l`
  mkdir -p $outdir/$category
  logfile=log-varmisuse-$category.txt
  rm -f $logfile
  for ((i=0;i<=$num;i++));
  do
     corr=$((i*2))
     incorr=$((corr+1))
     echo $corr,$incorr >> $logfile
     python3 gen_graph_jsons.py $correct_dir/file_$corr $misuse_dir/file_$incorr correct $outdir/$category/file_$corr.json varmisuse $pred_kind
     echo $? >> $logfile
     python3 gen_graph_jsons.py $misuse_dir/file_$incorr $correct_dir/file_$corr misuse $outdir/$category/file_$incorr.json varmisuse $pred_kind
     echo $? >> $logfile
done
}


function run_exception {
  category=$1
  indir=$2
  outdir=$3
  pred_kind=$4

  mkdir -p $outdir/$category
  logfile=log-exception-$category.txt
  rm -f $logfile
  for d in `ls -1 $indir/$category | grep -E "txt$"` ;
  do
    echo $d >> $logfile
    label=`cat $indir/$category/$d`
    IFS='_' read -ra splits <<< "$d" >& /dev/null
    name=${splits[0]}
    num=${splits[1]}
    python_file=$name'_'$num'.py'
    #$PYTHON_CONVERTOR -w $data_dir/$category/$python_file >& /dev/null
    python3 gen_graph_jsons.py $indir/$category/$python_file None $label $outdir/$category/graph_$python_file.json exception $pred_kind
    echo $? >> $logfile
  done
}

function run_defuse {
  category=$1
  indir=$2
  outdir=$3
  pred_kind=$4
  
  logfile=log-defuse-$category.txt
  rm -f $logfile
  for label in `ls -1 "$indir/$category"` ;
  do
    mkdir -p $outdir/$category
    for py_file in `ls -1 "$indir/$category/$label" | grep "py"` ;
    do
      echo $py_file >> $logfile
      python_file_path="$indir/$category/$label/$py_file"
      outpath="$outdir/$category/graph_$py_file.json"
      #$PYTHON_CONVERTOR -w -n $data_dir/$python_file >& /dev/null
      python gen_graph_jsons.py $python_file_path None $label $outpath defuse $pred_kind
      echo $? >> $logfile
    done
  done

}


## UNCOMMENT ANY OF THE FUNCTION BELOW TO GENERATE AST-BASED GRAPHS.

run_defuse dev /home/pardisp/defuse /home/pardisp/graphs_defuse prog_cls
# dev, eval, train

run_exception dev /home/pardisp/exception /home/pardisp/graphs_exception prog_cls
# dev, eval, train

run_varmisuse dev /home/pardisp/varmisuse /home/pardisp/graphs_varmisuse prog_cls
# dev, eval, train
