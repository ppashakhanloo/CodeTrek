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
grep "$bench/source.py" "var_misuses.txt" > "$outdir/edb_dumps_whitelist/var_misuses.csv" || exit 112345

echo "Generating Graph"
python build_graph.py "$outdir/edb_dumps_whitelist" python_full_table_joins.txt "$outdir/${fname}_graph"
echo "Graph Generated"

echo "Generating Walks"
python ../random_walk/gen_walks_misuse.py "$outdir/${fname}_graph.gv" "$outdir/edb_dumps_whitelist" "$label" "$outdir/walks.json"
echo "Walks Generated"
