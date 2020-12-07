#!/bin/bash

train_data_dir="$1"
test_data_dir="$2"
db_dir="$3"
classifier_script="$4"

rm -rf "$db_dir"
mkdir "$db_dir"

rm -rf source_root

rm -rf "edges"
mkdir edges
mkdir edges/train
mkdir edges/test

cp "$train_data_dir"/labels.txt edges/train/labels.txt
cp "$test_data_dir"/labels.txt edges/test/labels.txt

run_for_dir() { # arg: data_dir
  for f in `ls -1 "$1"`
  do
    # ignore txt files
    if [[ $f == *.txt ]]
    then
      continue
    fi
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
    cp "$db_dir"/graph-$f.edges edges/"$2"
    
    rm -rf source_root "$db_dir"
  done
}

run_for_dir "$train_data_dir" "train"
run_for_dir "$test_data_dir" "test"

# train
python3 "$classifier_script" edges
