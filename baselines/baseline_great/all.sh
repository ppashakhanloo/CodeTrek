#!/bin/bash

PYTHON_CONVERTOR=2to3-2.7

function run_varmisuse {
  category=$1 # dev, eval, train
  indir="$2" # py_files
  outdir="$3" # vm_graphs
  logfile=varmisuse-log-$category.txt

  rm -f $logfile
  mkdir -p $outdir/$category
  for source_file in `ls -1 "$indir/$category/correct"` ;
  do
     echo $source_file
     IFS='_' read -ra splits <<< "$source_files"
     IFS='.' read -ra splits <<< "${splits[1]}"
     corr=${splits[0]}
     incorr=$((splits[0]+1))
     outfile_corr=$outdir/$category/graph_file_$corr.py.json
     outfile_incorr=$outdir/$category/graph_file_$incorr.py.json
     #$PYTHON_CONVERTOR -w -n "$indir/$category/correct/file_$corr.py" >& /dev/null
     #$PYTHON_CONVERTOR -w -n "$indir/$category/misuse/file_$incorr.py" >& /dev/null
     echo $outfile_corr $outfile_incorr >> $logfile
     python gen_graph_jsons.py "$indir/$category/correct/file_$corr.py" "$indir/$category/misuse/file_$incorr.py" correct $outfile_corr varmisuse
     python gen_graph_jsons.py "$indir/$category/misuse/file_$incorr.py" "$indir/$category/correct/file_$corr.py" misuse $outfile_incorr varmisuse
     echo $? >> $logfile
  done
}

function run_exception {
  category=$1 # dev, eval, train
  indir="$2" # py_files
  outdir="$3" # exception_graphs
  logfile=exception-log-$category.txt

  rm -f $logfile
  mkdir -p $outdir/$category
  for label in `ls -1 "$indir/$category"` ;
  do
    for source_file in `ls -1 "$indir/$category/$label"` ;
    do
      echo $source_file
      py_file="$indir/$category/$label/$source_file"
      label=$label
      outfile=$outdir/$category/$label/graph_"$source_file".json
      #$PYTHON_CONVERTOR -w -n "$py_file" >& /dev/null
      echo $outfile >> $logfile
      python gen_graph_jsons.py "$py_file" None $label $outfile exception
      echo $? >> $logfile
    done
  done
}

function run_defuse {
  category=$1 # dev, eval, train
  indir="$2" # py_files
  outdir="$3" # defuse_graph
  logfile=defuse-log-$category.txt

  rm -f $logfile
  mkdir -p $outdir/$category
  for label in `ls -1 "$indir/$category"` ;
  do
    for source_file in `ls -1 "$indir/$category/$label"` ;
    do
      echo $source_file
      py_file="$indir/$category/$label/$source_file"
      outfile=$outdir/$category/graph_"$source_file".json
      #$PYTHON_CONVERTOR -w -n "$py_file" >& /dev/null
      echo $py_file >> $logfile
      python gen_graph_jsons.py "$py_file" None $label $outfile defuse
      echo $? >> $logfile
    done
  done
}

## UNCOMMENT ANY OF THE FUNCTION BELOW TO GENERATE GREAT GRAPHS.

run_defuse dev /home/pardisp/source_files/defuse /home/pardisp/great_graphs/defuse
#run_defuse eval /home/pardisp/source_files/defuse /home/pardisp/great_graphs/defuse
#run_defuse train /home/pardisp/source_files/defuse /home/pardisp/great_graphs/defuse

#run_exception dev /home/pardisp/source_files/exception-small /home/pardisp/great_graphs/exception-small
#run_exception eval /home/pardisp/source_files/exception-small /home/pardisp/great_graphs/exception-small
#run_exception train /home/pardisp/source_files/exception-small /home/pardisp/great_graphs/exception-small

#run_varmisuse dev /home/pardisp/source_files/varmisuse /home/pardisp/great_graphs/varmisuse
#run_varmisuse eval /home/pardisp/source_files/varmisuse /home/pardisp/great_graphs/varmisuse
#run_varmisuse train /home/pardisp/source_files/varmisuse /home/pardisp/great_graphs/varmisuse
