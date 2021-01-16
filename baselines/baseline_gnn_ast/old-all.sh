#!/bin/bash

correct_data_dir="$1"
incorrect_data_dir="$2"
classifier_script="$3"

rm -rf "edges"
mkdir edges
mkdir edges/correct
mkdir edges/incorrect

echo "Processing correct data dir..."
for f in `ls -1 "$correct_data_dir"`
do
  # run get_ast_edges to create edges file
  python3 get_ast_edges.py "$correct_data_dir/$f" edges/correct/graph-$f.edges
done

echo "Processing incorrect data dir..."
for f in `ls -1 "$incorrect_data_dir"`
do
  # run get_ast_edges to create edges file
  python3 get_ast_edges.py "$incorrect_data_dir/$f" edges/incorrect/graph-$f.edges
done

# train
python3 "$classifier_script" edges/correct edges/incorrect
