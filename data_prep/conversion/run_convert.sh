#!/bin/bash


kind=$1 # trek2gnn, trek2great
data_dir=$2
output_dir=$3

mkdir -p $output_dir

bin=""
if [ "$1" == "trek2gnn" ] ; then
  bin=convert_trek2gnn.py
elif [ "$1" == "trek2great" ] ; then
  bin=convert_trek2great.py
fi

for f in `ls -1 $data_dir | grep graph_`
do
  echo "$f"
  graph_file=$data_dir/$f
  stb=stub${f:5:-2}json
  stub_file=$data_dir/$stb
  out_file=$output_dir/graph${f:5:-2}json

  python3 $bin $graph_file $stub_file $out_file
done
