#!/bin/bash

dir=$1

echo -e "filename\tnodes\tedges\tvars"

for f in `ls -1 "$dir"`
do
  echo -e "$f\t`gvpr -f basic_stats.gvpr $dir/$f`\t`grep -c 'variable(' $dir/$f`"
done


