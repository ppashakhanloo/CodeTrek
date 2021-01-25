#!/bin/bash

gv_file=testfiles/example.gv
out_file=example.json

python3 gen_walks.py $gv_file $out_file
