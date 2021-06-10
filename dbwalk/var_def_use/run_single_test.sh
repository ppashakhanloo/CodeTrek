#!/bin/bash

py_source="test.py"
saved_model=""
lang=python

save_dir=$HOME/scratch/results/dbwalk/gen
queries_dir=$HOME/"relational-representation/python_codeql/python-edb-queries/queries"
codeql_runner=$HOME/"relational-representation/codeql_dir/codeql/codeql"
defuse_stub_bin=$HOME/"relational-representation/data_prep/random_walk/gen_stubs_defuse.py"
graph_bin=$HOME/"relational-representation/data_prep/graph/build_graph.py"
join_path=$HOME/"relational-representation/python_codeql/join.txt"
cook_bin=$HOME/"relational-representation/dbwalk/data_util/cook_single_gv_stub.py"

tmp_dir=$(mktemp -d)
bqrs_dir=$tmp_dir/"db/results/python-edb-queries/"
tables_dir=$tmp_dir/tables
proc_dir=$tmp_dir/data

mkdir -p $tmp_dir $proc_dir $tables_dir
cp $py_source $tmp_dir
$codeql_runner database create $tmp_dir/db --language=python --source-root=$tmp_dir
$codeql_runner database run-queries $tmp_dir/db $queries_dir
for bqrs in `ls -1 $bqrs_dir | grep bqrs` ;
do
  $codeql_runner bqrs decode --entities=id --output=$tables_dir/$bqrs.csv --format=csv $bqrs_dir/$bqrs
done

python $graph_bin $tables_dir $join_path $proc_dir/graph_$py_source
python $defuse_stub_bin $proc_dir/graph_$py_source.gv $tables_dir $tables_dir/unused_var.bqrs.csv $proc_dir/stub_$py_source.json

python -m dbwalk.data_util.cook_single_gv_stub \
    -language $lang \
    -single_source $proc_dir/graph_$py_source.gv \
    -use_node_val True \
    -online_walk_gen True \
    $@

bsize=1
embed=256
nlayer=4
nhead=8
hidden=512
dropout=0
setenc=deepset
online=True
num_proc=0
shuffle_var=False
use_node_val=True

export CUDA_VISIBLE_DEVICES=0

python main.py \
    -save_dir $saved_model \
    -data $data_name \
    -online_walk_gen $online \
    -set_encoder $setenc \
    -shuffle_var $shuffle_var \
    -batch_size $bsize \
    -embed_dim $embed \
    -nhead $nhead \
    -transformer_layers $nlayer \
    -dim_feedforward $hidden \
    -dropout $dropout \
    -iter_per_epoch 1000 \
    -num_proc $num_proc \
    -use_node_val $use_node_val \
    -learning_rate 1e-4 \
    -min_steps 4 \
    -max_steps 20 \
    -gpu 0 \
    $@



