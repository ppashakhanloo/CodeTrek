#!/bin/bash

# varmisuse prog_cls
python build_ast_for_codetrek.py varmisuse tables cq_ast_prog_cls testfiles/path_vm.txt varmisuse prog_cls

# varmisuse loc_cls
python build_ast_for_codetrek.py varmisuse tables cq_ast_loc_cls testfiles/path_vm.txt varmisuse loc_cls

# exception
python build_ast_for_codetrek.py exception-small tables cq_ast_prog_cls testfiles/path_ex.txt exception prog_cls

# defuse prog_cls
python build_ast_for_codetrek.py defuse tables cq_ast_prog_cls testfiles/path_du.txt defuse prog_cls

# defuse loc_cls
python build_ast_for_codetrek.py defuse tables cq_ast_loc_cls testfiles/path_du.txt defuse loc_cls

