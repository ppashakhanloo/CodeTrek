#!/bin/bash

dir="$1"

for file in `ls -1 "$dir"`
do
  if [[ "$file" == *csv ]]
  then
    tail -n +2 "$dir/$file" > "$dir/$file.facts"
  fi
done
