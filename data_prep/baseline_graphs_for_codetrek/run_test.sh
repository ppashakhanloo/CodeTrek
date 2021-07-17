#!/bin/bash

# varmisuse prog_cls
rm -f stub_testfile1.py.json graph_testfile2.py.gv
python build_ast_for_codetrek.py testfiles/varmisuse/dev/correct/testfile1.py testfiles/varmisuse/dev/misuse/testfile2.py varmisuse prog_cls

# exception
rm -f stub_testfile3.py.json graph_testfile3.py.gv
python build_ast_for_codetrek.py testfiles/exception/dev/KeyError/testfile3.py none exception prog_cls

# defuse prog_cls
rm -f stub_testfile4.py.json graph_testfile4.py.gv
python build_ast_for_codetrek.py testfiles/defuse/dev/unused/testfile4.py none defuse prog_cls

