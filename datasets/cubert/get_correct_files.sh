#!/usr/bin/env bash

filename="$1"

for i in $(cat "$filename")
do
    num=$(basename $(dirname $i))
    num=${num:5}
    label=$(basename $(dirname $(dirname $i)))
    base_dir=$(dirname $(dirname $(dirname $i)))
    
    if [[ "$label" == "misuse" ]]
    then
        echo $base_dir/correct/file_$((num-1))/source.py
    else
        echo $i
    fi
done