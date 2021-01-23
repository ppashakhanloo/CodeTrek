#!/usr/bin/env bash

echo "file1 , var1 , start_row1 , start_col1 , end_row1 , end_col1 , file2 , var2 , start_row2 , start_col2 , end_row2 , end_col2"

for correct_file in $(cat "$1")
do
    num=$(basename $(dirname $correct_file))
    num=${num:5}
    dir=$(dirname $(dirname $(dirname $correct_file)))
    misuse_file="$dir/misuse/file_$((num+1))/source.py"
    # echo $correct_file $misuse_file
    python diff.py $correct_file $misuse_file || exit 123 
done
