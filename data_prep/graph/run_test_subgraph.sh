#!/bin/bash

graph_file=../random_walk/testfiles/train/graph_exception_1.gv
stub_file=../random_walk/testfiles/train/stub_exception_1.json
node_num=500
out_file=subgraph.gv

python3 sample_graph.py $graph_file $stub_file $node_num $out_file

