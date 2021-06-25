#!/bin/bash

# test random walk gen
graph_file=testfiles/train/graph_exception_1.gv
json_file=testfiles/train/stub_exception_1.json
#python test_walkgen.py $graph_file $json_file

##### test stub generator -- varmisuse #####
gv1=testfiles/stub/graph_file_0.py.gv
gv2=testfiles/stub/graph_file_1.py.gv
edb_path=testfiles/stub/tables
stub_file=testfiles/stub/vm_stub.json
py1=testfiles/stub/file1.py
py2=testfiles/stub/file2.py

echo "Run diff.py to generate var_misuses:"
python diff.py $py1 $py2 $edb_path/var_misuses.csv
echo "Finished with status" $?

echo "Testing gen_stubs_varmisuse.py:"
python gen_stubs_varmisuse.py $gv1 $edb_path correct $stub_file"_prog_g" graphs prog_cls
echo "prog_cls, graphs:" $?
python gen_stubs_varmisuse.py $gv1 $edb_path correct $stub_file"_loc_g" graphs loc_cls
echo "loc_cls, graphs:" $?
python gen_stubs_varmisuse.py $gv1 $edb_path correct $stub_file"_prog_w" walks prog_cls
echo "proc_cls, walks:" $?
python gen_stubs_varmisuse.py $gv1 $edb_path correct $stub_file"_loc_w" walks loc_cls
echo "loc_cls, walks:" $?


##### test stub generator -- defuse #####
gv=testfiles/stub/graph_file_2.py.gv
edb_path=testfiles/stub/tables2
stub_file=testfiles/stub/du_stub.json

echo "Testing gen_stubs_defuse.py:"
python gen_stubs_defuse.py $gv $edb_path $stub_file"_prog_g" graphs prog_cls
echo "prog_cls, graphs:" $?
python gen_stubs_defuse.py $gv $edb_path $stub_file"_loc_g" graphs loc_cls
echo "loc_cls, graphs:" $?
python gen_stubs_defuse.py $gv $edb_path $stub_file"_prog_w" walks prog_cls
echo "proc_cls, walks:" $?
python gen_stubs_defuse.py $gv $edb_path $stub_file"_loc_w" walks loc_cls
echo "loc_cls, walks:" $?

exit 0

##### test stub generator -- exception #####
gv=testfiles/stub/graph_file_3.py.gv
edb_path=testfiles/stub/tables3
stub_file=testfiles/stub/ex_stub.json

echo "Testing gen_stubs_exception.py:"
python gen_stubs_exception.py $gv $edb_path correct $stub_file"_prog_g" graphs prog_cls
echo "prog_cls, graphs:" $?
python gen_stubs_exception.py $gv $edb_path correct $stub_file"_loc_g" graphs loc_cls
echo "loc_cls, graphs:" $?
python gen_stubs_exception.py $gv $edb_path correct $stub_file"_prog_w" walks prog_cls
echo "proc_cls, walks:" $?
python gen_stubs_exception.py $gv $edb_path correct $stub_file"_loc_w" walks loc_cls
echo "loc_cls, walks:" $?


