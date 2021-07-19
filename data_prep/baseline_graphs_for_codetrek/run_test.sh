#!/bin/bash

rm -f stub_* graph_*

# varmisuse prog_cls correct
python build_ast_for_codetrek.py testfiles/varmisuse/dev/correct/testfile1.py testfiles/varmisuse/dev/misuse/testfile2.py varmisuse prog_cls
mv stub_testfile1.py.json stub_vm_prog_correct.json

# varmisuse loc_cls correct
python build_ast_for_codetrek.py testfiles/varmisuse/dev/correct/testfile1.py testfiles/varmisuse/dev/misuse/testfile2.py varmisuse loc_cls
mv stub_testfile1.py.json stub_vm_loc_correct.json

# varmisuse prog_cls misuse
python build_ast_for_codetrek.py testfiles/varmisuse/dev/misuse/testfile2.py testfiles/varmisuse/dev/correct/testfile1.py varmisuse prog_cls
mv stub_testfile2.py.json stub_vm_prog_misuse.json

# varmisuse loc_cls misuse
python build_ast_for_codetrek.py testfiles/varmisuse/dev/misuse/testfile2.py testfiles/varmisuse/dev/correct/testfile1.py varmisuse loc_cls
mv stub_testfile2.py.json stub_vm_loc_misuse.json


# exception
python build_ast_for_codetrek.py testfiles/exception/dev/KeyError/testfile3.py none exception prog_cls
mv stub_testfile3.py.json stub_ex_prog.json


# defuse prog_cls unused
python build_ast_for_codetrek.py testfiles/defuse/dev/unused/testfile4.py none defuse prog_cls
mv stub_testfile4.py.json stub_du_prog_unused.json

# defuse prog_cls used
python build_ast_for_codetrek.py testfiles/defuse/dev/used/testfile5.py none defuse prog_cls
mv stub_testfile5.py.json stub_du_prog_used.json

# defuse loc_cls
python build_ast_for_codetrek.py defuse tables defuse/dev/unused/file_1129004.py none defuse loc_cls

