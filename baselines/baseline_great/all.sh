#!/bin/bash

function run_varmisuse {
  category=$1 # dev, eval, train
  indir="$2" # py_files
  outdir="$3" # vm_graphs
  
  num=`ls -1 "$indir/$category/correct" | wc -l` 
  for ((i=0;i<=$num;i++));
  do # file_number.py
     corr=$((i*2))
     incorr=$((corr+1))
     outfile_corr=$outdir/$category/correct/graph_file_$corr.py.json
     outfile_incorr=$outdir/$category/misuse/graph_file_$incorr.py.json
     python gen_graph_jsons.py "$indir/$category/correct/file_$corr.py" "$indir/$category/misuse/file_$incorr.py" correct $outfile_corr varmisuse
     python gen_graph_jsons.py "$indir/$category/misuse/file_$incorr.py" "$indir/$category/correct/file_$corr.py" misuse $outfile_incorr varmisuse
  done
}


function run_exception {
  indir="$1" # py_files
  outdir="$2" # exception_graphs

  for category in `ls -1 "$indir"` ;
  do
    for exception in `ls -1 "$indir/$category"` ;
    do
      mkdir -p $outdir/$category/$exception
      for source_file in `ls -1 "$indir/$category/$exception"` ;
      do
        py_file="$indir/$category/$exception/$source_file"
        label=$exception
        outfile=$outdir/$category/$exception/graph_"$source_file".json
        python gen_graph_jsons.py "$py_file" None $label $outfile exception
      done
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
      python gen_graph_jsons.py "$py_file" None $label $outfile defuse
    done
  done
}


## UNCOMMENT ANY OF THE FUNCTION BELOW TO GENERATE AST-BASED GRAPHS.

#run_defuse dev pys_dir out_graphs
#run_defuse eval pys_dir out_graphs
#run_defuse train pys_dir out_graphs

#run_exception pys_dir out_graphs

#run_varmisuse dev pys_dir out_graphs
#run_varmisuse eval pys_dir out_graphs
#run_varmisuse train pys_dir out_graphs
