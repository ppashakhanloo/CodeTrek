#!/bin/bash

start=0
end=100
width=100
n=0

rm -rf allwalks
mkdir allwalks

benchdir="../datasets/cubert/py_files"

filelist="$1"

total=$(cat "$filelist" | wc -l)
echo $total

for p in $(cat "$filelist")
do
    if (($n >= $start && $n < $end))
    then
        bench="../datasets/cubert/$(dirname $p)"
        echo "Running for $bench"
        ./getwalks_misuse.sh "$bench" > /dev/null &
        echo "$n / $total added till now"
    else
        echo "Waiting for all background processes to finish"
        wait
        echo "Background processes finished"
        start=$end
        end=$((end+width))
        echo "Setting start and end to $start and $end"
        bench="../datasets/cubert/$(dirname $p)"
        echo "Running for $bench"
        ./getwalks_misuse.sh "$bench" > /dev/null &
        echo "$n / $total added till now"
    fi
    n=$((n+1))
done
wait
