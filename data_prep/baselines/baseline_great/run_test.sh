#!/bin/bash

rm -rf output_files
mkdir output_files

## VAR MISUSE TASK ##
python gen_graph_jsons.py testfiles/correct.py testfiles/incorrect.py correct output_files/output_1.json varmisuse
python gen_graph_jsons.py testfiles/incorrect.py testfiles/correct.py misuse output_files/output_2.json varmisuse

## EXCEPTION TASK ###
python gen_graph_jsons.py testfiles/exception.py None KeyError output_files/output_ex.json exception

## DEF-USE TASK #####
python gen_graph_jsons.py testfiles/defuse.py None used output_files/output_defuse.json defuse
