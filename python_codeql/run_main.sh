#!/bin/bash

## set the following two lines:
DATA_SOURCE_BASE=/home/pardisp/raw_source_files
IMPLEMENTATION_BASE=/home/pardisp/relational-representation



GEN_STUB_BASE=$IMPLEMENTATION_BASE/data_prep/random_walk
TASK_QUERIES_BASE=$IMPLEMENTATION_BASE/python_codeql/python-edb-queries
GRAPH_BUILDER=$IMPLEMENTATION_BASE/data_prep/graph/build_graph.py

run_one_dir_exception() {
  category="$1" # dev, eval, train
  size=$2 # small, large
  data_dir="$DATA_SOURCE_BASE/$category"  # example: /home/pardisp/raw_source_data/dev/{.py,_label.txt}
  db_dir=$DATA_SOURCE_BASE"-db"/$category
  tmp_filename=tmp_src_el_$category
  query_dir=$TASK_QUERIES_BASE/task_exception_queries_$size
  log_filename=log_el_$category.txt

  rm -rf $log_filename $tmp_filename
  mkdir -p $db_dir

  for f in `ls -1 $data_dir | grep -E "\.py$"` ;
  do
    echo Processing $data_dir/$f
    echo $data_dir/$f >> $log_filename

    echo "generate database..."
    rm -rf $tmp_filename
    mkdir $tmp_filename
    cp $data_dir/$f $tmp_filename/
    codeql database create $db_dir/$f --threads=32 --language=python --quiet --source-root=$tmp_filename >& /dev/null
    echo "gen db" $? >> $log_filename

    echo "run the queries..."
    ./extract_all.sh $query_dir $db_dir/$f $db_dir/$f/tables 32 >& /dev/null
    echo "run qs" $? >> $log_filename
    
    echo "build graph..."
    python3 $GRAPH_BUILDER $db_dir/$f/tables python_full_table_joins.txt $db_dir/$f/graph_$f
    echo "graph" $? >> $log_filename

    echo "gen stubs..."
    python3 $GEN_STUB_BASE/gen_stubs_exception.py $db_dir/$f/graph_$f.gv $db_dir/$f/tables $data_dir/"$f"_label.txt $db_dir/$f/stub_$f.json
    echo "stub" $? >> $log_filename
  done
}

run_one_dir_misuse() {
  category="$1" # dev, eval, train
  label="$2" # correct, misuse
  data_dir="$DATA_SOURCE_BASE/$category" # example: /home/pardisp/raw_source_data/dev/{correct,misuse}/{.py}
  db_dir=$DATA_SOURCE_BASE"-db"/$category
  tmp_filename=tmp_src_vmu_$category
  query_dir=$TASK_QUERIES_BASE/task_varmisuse_queries
  log_filename=log_vmu_$category.txt

  rm -rf $log_filename $tmp_filename
  mkdir -p $db_dir

  for f in `ls -1 $data_dir | grep -E "\.py$"` ;
  do
    echo Processing $data_dir/$f
    echo $data_dir/$f >> $log_filename

    echo "generate database..."
    rm -rf $tmp_filename
    mkdir $tmp_filename
    cp $data_dir/$f $tmp_filename/
    codeql database create $db_dir/$f --threads=32 --language=python --quiet --source-root=$tmp_filename >& /dev/null
    echo "gen db" $? >> $log_filename

    echo "run the queries..."
    ./extract_all.sh $query_dir $db_dir/$f $db_dir/$f/tables 32 >& /dev/null
    echo "run qs" $? >> $log_filename
    
    echo "build graph..."
    python3 $GRAPH_BUILDER $db_dir/$f/tables python_full_table_joins.txt $db_dir/$f/graph_$f
    echo "graph" $? >> $log_filename

    echo "gen stubs..."
    python3 $GEN_STUB_BASE/gen_stubs_varmisuse.py $db_dir/$f/graph_$f.gv $label $db_dir/$f/stub_$f.json
    echo "stub" $? >> $log_filename
  done
}

run_one_dir_defuse() {
  category="$1" # dev, eval, train
  data_dir="$DATA_SOURCE_BASE/$category" # example: /home/pardisp/raw_source_data/dev/{.py}
  db_dir="$DATA_SOURCE_BASE"-db/$category
  tmp_filename=tmp_src_du_$category
  query_dir=$TASK_QUERIES_BASE/task_defuse_queries
  log_filename=log_du_$category.txt

  rm -rf $log_filename $tmp_filename
  mkdir -p $db_dir

  for f in `ls -1 $data_dir | grep -E "\.py$"` ;
  do
    echo Processing $data_dir/$f
    echo $data_dir/$f >> $log_filename

    echo "generate database..."
    rm -rf $tmp_filename
    mkdir $tmp_filename
    cp $data_dir/$f $tmp_filename/
    codeql database create $db_dir/$f --threads=32 --language=python --quiet --source-root=$tmp_filename >& /dev/null
    echo "gen db" $? >> $log_filename

    echo "run the queries..."
    ./extract_all.sh $query_dir $db_dir/$f $db_dir/$f/tables 32 >& /dev/null
    echo "run qs" $? >> $log_filename
    
    echo "build graph..."
    python3 $GRAPH_BUILDER $db_dir/$f/tables python_full_table_joins.txt $db_dir/$f/graph_$f
    echo "graph" $? >> $log_filename

    echo "gen stubs..."
    python3 $GEN_STUB_BASE/gen_stubs_defuse.py $db_dir/$f/graph_$f.gv $db_dir/$f/tables $db_dir/$f/tables/unused_var.csv $db_dir/$f/stub_$f.json
    echo "stub" $? >> $log_filename
  done
}



############## ENTRY POINT ##############
if [ "$1" == "exception_large" ] ; then
    run_one_dir_exception "dev" "large"
    run_one_dir_exception "eval" "large"
    run_one_dir_exception "train" "large"
elif [ "$1" == "exception_small" ] ; then
    run_one_dir_exception "dev" "small"
    run_one_dir_exception "eval" "small"
    run_one_dir_exception "train" "small"
elif [ "$1" == "defuse" ] ; then
    run_one_dir_defuse "dev"
    run_one_dir_defuse "eval"
    run_one_dir_defuse "train"
elif [ "$1" == "varmisuse" ] ; then
    run_one_dir_varmisuse "dev" "correct"
    run_one_dir_varmisuse "eval" "correct"
    run_one_dir_varmisuse "train" "correct"

    run_one_dir_varmisuse "dev" "misuse"
    run_one_dir_varmisuse "eval" "misuse"
    run_one_dir_varmisuse "train" "misuse"
fi
########################################


