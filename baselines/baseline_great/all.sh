#!/bin/bash

PYTHON_CONVERTOR=2to3-2.7

function run_varmisuse {
  category=$1 # dev, eval, train
  indir="$2" # py_files
  outdir="$3" # vm_graphs
  
  mkdir -p $outdir/$category/correct
  mkdir -p $outdir/$category/misuse
  for vm in `ls -1 "$indir/$category/correct"` ;
  do
     echo $vm
     IFS='_' read -ra splits <<< "$vm"
     IFS='.' read -ra splits <<< "${splits[1]}"
     corr=${splits[0]}
     incorr=$((splits[0]+1))
     outfile_corr=$outdir/$category/correct/graph_file_$corr.py.json
     outfile_incorr=$outdir/$category/misuse/graph_file_$incorr.py.json
     $PYTHON_CONVERTOR -w -n "$indir/$category/correct/file_$corr.py" >& /dev/null
     $PYTHON_CONVERTOR -w -n "$indir/$category/misuse/file_$incorr.py" >& /dev/null
     python gen_graph_jsons.py "$indir/$category/correct/file_$corr.py" "$indir/$category/misuse/file_$incorr.py" correct $outfile_corr varmisuse
     python gen_graph_jsons.py "$indir/$category/misuse/file_$incorr.py" "$indir/$category/correct/file_$corr.py" misuse $outfile_incorr varmisuse
  done
}

function run_exception {
  category=$1 # dev, eval, train
  indir="$2" # py_files
  outdir="$3" # exception_graphs

  for exception in `ls -1 "$indir/$category"` ;
  do
    mkdir -p $outdir/$category/$exception
    for source_file in `ls -1 "$indir/$category/$exception"` ;
    do
      echo $source_file
      py_file="$indir/$category/$exception/$source_file"
      label=$exception
      outfile=$outdir/$category/$exception/graph_"$source_file".json
      $PYTHON_CONVERTOR -w -n "$py_file" >& /dev/null
      python gen_graph_jsons.py "$py_file" None $label $outfile exception
    done
  done
}

function run_defuse {
  category=$1 # dev, eval, train
  indir="$2" # py_files
  outdir="$3" # defuse_graph

  for label in `ls -1 "$indir/$category"` ;
  do
    mkdir -p $outdir/$category/$label
    for source_file in `ls -1 "$indir/$category/$label"` ;
    do
      py_file="$indir/$category/$label/$source_file"
      outfile=$outdir/$category/$label/graph_"$source_file".json
      $PYTHON_CONVERTOR -w -n "$py_file" >& /dev/null
      python gen_graph_jsons.py "$py_file" None $label $outfile defuse
    done
  done
}

## UNCOMMENT ANY OF THE FUNCTION BELOW TO GENERATE AST-BASED GRAPHS.

#run_defuse dev pys_dir out_graphs
#run_defuse eval pys_dir out_graphs
#run_defuse train pys_dir out_graphs

run_exception dev /home/pardisp/source_files/exception-small /home/pardisp/great_graphs/exception-small
#run_exception eval /home/pardisp/source_files/exception-small /home/pardisp/great_graphs/exception-small
#run_exception train /home/pardisp/source_files/exception-small /home/pardisp/great_graphs/exception-small

#run_varmisuse dev /home/pardisp/source_files/varmisuse /home/pardisp/great_graphs/varmisuse
#run_varmisuse eval /home/pardisp/source_files/varmisuse /home/pardisp/great_graphs/varmisuse
#run_varmisuse train /home/pardisp/source_files/varmisuse /home/pardisp/great_graphs/varmisuse
