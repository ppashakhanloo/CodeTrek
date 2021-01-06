#!/bin/bash

bench="$1"

rm -rf ${bench}/edb_dumps_whitelist
mkdir -p ${bench}/edb_dumps_whitelist
rm -rf ${bench}/results
mkdir -p ${bench}/results
mkdir -p ${bench}/cluster_logs
rm -rf ${bench}/cluster_logs/dumplog_whitelist.*
timeout 30m ./extract_all.sh python-edb-queries/whitelist-queries ${bench}/codeqldb ${bench}/edb_dumps_whitelist 8 > ${bench}/cluster_logs/dumplog_whitelist.out 2>${bench}/cluster_logs/dumplog_whitelist.err
err="$?"
if [[ $err != 0 ]]
then
    echo "ERROR $err" >> ${bench}/cluster_logs/dumplog_whitelist.err
fi