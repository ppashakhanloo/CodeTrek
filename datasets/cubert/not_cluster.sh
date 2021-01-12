#!/usr/bin/env bash

bench="$1"
mkdir -p ${bench}/cluster_logs
rm -rf ${bench}/codeqldb
codeql database create ${bench}/codeqldb --language=python --source-root=${bench} > ${bench}/cluster_logs/createlog.out 2>${bench}/cluster_logs/createlog.err
codeql query run -d ${bench}/codeqldb -o ${bench}/codeqldb/unused_local_vars.bqrs unusedlocalvar.ql > ${bench}/cluster_logs/runlog.out 2>${bench}/cluster_logs/runlog.err
codeql bqrs decode -o ${bench}/codeqldb/unused_local_vars.csv --format=csv ${bench}/codeqldb/unused_local_vars.bqrs > ${bench}/cluster_logs/decodelog.out 2>${bench}/cluster_logs/decodelog.err
codeql bqrs decode -o ${bench}/codeqldb/unused_local_vars_id.csv --entities=id --format=csv ${bench}/codeqldb/unused_local_vars.bqrs > ${bench}/cluster_logs/decodeidlog.out 2>${bench}/cluster_logs/decodeidlog.err
rm -rf ${bench}/edb_dumps_whitelist
mkdir -p ${bench}/edb_dumps_whitelist
rm -rf ${bench}/results
mkdir -p ${bench}/results
mkdir -p ${bench}/cluster_logs
rm -rf ${bench}/cluster_logs/dumplog.*
../../python_codeql/extract_all.sh ../../python_codeql/python-edb-queries/whitelist-queries ${bench}/codeqldb ${bench}/edb_dumps_whitelist 8 > ${bench}/cluster_logs/dumplog.out 2>${bench}/cluster_logs/dumplog.err
