#!/bin/bash

correct_data_dir="$1"
incorrect_data_dir="$2"
db_dir="$3"
classifier_script="$4"

rm -rf "$db_dir"
mkdir "$db_dir"
rm -rf source_root

rm -rf "edges"
mkdir edges
mkdir edges/correct
mkdir edges/incorrect

run_for_dir() { # arg: data_dir
  for f in `ls -1 "$1"`
  do
    # make source-root
    mkdir source_root
    cp "$1"/"$f" source_root/"$f"
  
    # create codeql database
    codeql database create "$db_dir" --language=python --source-root=source_root
  
    # run queries to extract tables of interest
    ./extract_all.sh python-edb-queries/edb-queries "$db_dir" "$db_dir" 0

    # run tables to facts to remove first line (header)
    ./remove_header.sh "$db_dir"
  
    # run build_graph to create edges file
    python3 build_graph.py "$db_dir" python_full_table_joins.txt "$db_dir"/graph-$f

    # copy the edges
    cp "$db_dir"/graph-$f.edges edges/correct/
    rm -rf source_root "$db_dir"
  done
}

run_for_dir "$correct_data_dir"
run_for_dir "$incorrect_data_dir"

# train
python3 "$classifier_script" edges/correct edges/incorrect
