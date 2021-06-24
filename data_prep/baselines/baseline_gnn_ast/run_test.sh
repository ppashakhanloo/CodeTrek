#!/bin/bash

rm -rf output_files
mkdir output_files

## VAR MISUSE TASK ##
python3 gen_graph_jsons.py testfiles/correct.py testfiles/incorrect.py correct output_files/output_1.json varmisuse loc_cls
python3 gen_graph_jsons.py testfiles/incorrect.py testfiles/correct.py misuse output_files/output_2.json varmisuse loc_cls

## EXCEPTION TASK ###
python3 gen_graph_jsons.py testfiles/exception.py None KeyError output_files/output_ex.json exception prog_cls

## DEF-USE TASK #####
python3 gen_graph_jsons.py testfiles/defuse.d.number.py None unused output_files/output_defuse.json defuse prog_cls
