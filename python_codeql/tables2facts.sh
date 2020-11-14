#!/usr/local/bin/bash

dir=$1

for file in `ls -1 $dir`
do
  tail -n +2 "$dir/$file" > "$dir/temp"
  cat "$dir/temp" > "$dir/$file"
  #sed 's/,/	/g' "$dir/temp" > "$dir/$file.facts"
  rm -f "$dir/temp"
done
