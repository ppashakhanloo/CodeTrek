#!/bin/bash

if [[ $# -ne 1 ]]
then
    echo "Usage: ./get_edb.sh <db-dir>"
    exit 1
fi

mkdir -p edb-$1
for edb_query in $(ls cpp-edb-queries)
do
    codeql query run --database="$1" --output="$edb_query.bqrs" "cpp-edb-queries/$edb_query"
    codeql bqrs decode --output="edb-$1/${edb_query%.*}.csv" --format=csv --entities=id "$edb_query.bqrs"
    rm "$edb_query.bqrs"
done
