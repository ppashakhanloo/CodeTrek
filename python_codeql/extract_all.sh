#!/bin/bash

if [ $# -ne 4 ]
then
    echo "Usage ./extract_all.sh <query-dir> <db-dir> <out-dir> <num-threads>"
    exit 1
fi

dir="$1"
db="$2"
out="$3"
threads="$4"

# Generate the BQRS files
codeql database run-queries --threads="$threads" $db $dir

# Generate CSV files
bqrs_dir="$db/results"
mkdir -p $out
echo "Decoding all .bqrs files..."
for path in $(find $bqrs_dir -name "*.bqrs")
do
    f="$(basename $path)"
    # echo "Decoding for" $f
   { codeql bqrs decode --output="$out/${f%.*}".csv --format=csv --entities=id "$path" && rm -rf path; } &
done
echo "Waiting for all jobs to finish"
wait
rm -rf $bqrs_dir
echo "Done"

