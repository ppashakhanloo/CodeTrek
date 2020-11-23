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

for f in `ls -1 "$correct_data_dir"`
do
  # make source-root
  mkdir source_root
  cp "$correct_data_dir"/"$f" source_root/"$f"
  
  # create codeql database
  codeql database create "$db_dir"/correct-"$f" --language=python --source-root=source_root
  
  # run queries to extract tables of interest
  ./extract_py_tables.sh "$db_dir"/correct-"$f" "$db_dir"/correct-"$f"

  # run tables to facts to remove first line (header)
  ./tables2facts.sh "$db_dir"/correct-"$f"
  
  # run build_graph to create edges file
  python3 build_graph.py "$db_dir"/correct-"$f" table_joins.txt "$db_dir"/correct-"$f"/graph-$f

  rm -rf source_root

  # copy the edges
  cp "$db_dir"/correct-"$f"/graph-$f.edges edges/correct/
done

for f in `ls -1 "$incorrect_data_dir"`
do
  # make source-root
  mkdir source_root
  cp "$incorrect_data_dir"/"$f" source_root/"$f"
  
  # create codeql database
  codeql database create "$db_dir"/incorrect-"$f" --language=python --source-root=source_root
  
  # run queries to extract tables of interest
  ./extract_py_tables.sh "$db_dir"/incorrect-"$f" "$db_dir"/incorrect-"$f"

  # run tables to facts to remove first line (header)
  ./tables2facts.sh "$db_dir"/incorrect-"$f"
  
  # run build_graph to create edges file
  python3 build_graph.py "$db_dir"/incorrect-"$f" table_joins.txt "$db_dir"/incorrect-"$f"/graph-$f

  rm -rf source_root

  # copy the edges
  cp "$db_dir"/incorrect-"$f"/graph-$f.edges edges/incorrect/
done

# train
python3 "$classifier_script" edges/correct edges/incorrect
