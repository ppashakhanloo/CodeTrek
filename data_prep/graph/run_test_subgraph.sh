#!/bin/bash

data_dir=$1
node_num=10000
output_dir=$2

mkdir -p $output_dir

for f in `ls -1 $data_dir | grep graph_`
do
  echo "$f"
  graph_file=$data_dir/$f
  stb=stub${f:5:-2}json
  stub_file=$data_dir/$stb
  out_file=$output_dir/$f
  python3 sample_graph.py $graph_file $stub_file $node_num $out_file
  cp $stub_file $output_dir
done
