#!/usr/bin/env bash

bench="$1"

mkdir -p ${bench}/cluster_logs  
rm -rf ${bench}/codeqldb/unused_local_vars*
codeql query run -d ${bench}/codeqldb -o ${bench}/codeqldb/unused_local_vars.bqrs unusedlocalvar.ql > ${bench}/cluster_logs/runlog.out 2>${bench}/cluster_logs/runlog.err
codeql bqrs decode -o ${bench}/codeqldb/unused_local_vars.csv --format=csv ${bench}/codeqldb/unused_local_vars.bqrs > ${bench}/cluster_logs/decodelog.out 2>${bench}/cluster_logs/decodelog.err
codeql bqrs decode -o ${bench}/codeqldb/unused_local_vars_id.csv --entities=id --format=csv ${bench}/codeqldb/unused_local_vars.bqrs > ${bench}/cluster_logs/decodeidlog.out 2>${bench}/cluster_logs/decodeidlog.err
