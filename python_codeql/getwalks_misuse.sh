#!/usr/bin/env bash

if [[ $# != 1 ]]
then
    echo "Usage: ./getwalks.sh path/to/bench/dir"
    exit 1
fi

bench="$1"
dname=$(dirname "$bench")
label=$(basename $dname)
set=$(basename $(dirname $dname))
fname=$(basename "$bench")
outdir="allwalks/walks_${set}_$fname"

rm -rf "$outdir"
mkdir -p "$outdir"

cp -r "$bench/edb_dumps_whitelist" "$bench/source.py" "$outdir"
grep "$bench" "var_misuses.txt" > "$outdir/var_misuses.csv"

echo "Generating Graph"
python build_graph.py "$outdir/edb_dumps_whitelist" python_full_table_joins.txt "$outdir/${fname}_graph"
echo "Graph Generated"

echo "Generating Walks"
python ../random_walk/gen_walks_misuse.py "$outdir/${fname}_graph.gv" "$outdir/edb_dumps_whitelist" "$outdir/unused_local_vars_id.csv" "$outdir/walks.json"
echo "Walks Generated"
