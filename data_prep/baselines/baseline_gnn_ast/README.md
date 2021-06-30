
## Test the datapoint
``
python3 datapoint_test.py 
``

## AST graph generators
``
./run_varmisuse.sh splits varmisuse /home/pardisp/relational-representation prog_cls
./run_varmisuse.sh splits varmisuse /home/pardisp/relational-representation loc_cls
./run_varmisuse.sh splits varmisuse /home/pardisp/relational-representation loc_rep
``

``
./run_defuse.sh splits defuse /home/pardisp/relational-representation prog_cls
./run_defuse.sh splits defuse /home/pardisp/relational-representation loc_cls
``

``
./run_exception.sh splits exception-small /home/pardisp/relational-representation
``
