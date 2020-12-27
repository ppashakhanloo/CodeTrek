#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Usage ./remove_header.sh <csv-dir>"
    exit 1
fi

dir="$1"

for file in `ls -1 "$dir"`
do
  if [[ "$file" == *csv ]]
  then
    tail -n +2 "$dir/$file" > "$dir/$file.facts"
  fi
done
