#!/usr/local/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage ./extract_py_tables.sh <query-dir> <db-dir>"
    exit 1
fi

dir="$1"
db="$2"

declare -A headers
headers["ClassObject"]="id,inferredType"
headers["Location"]="id,beginLine,beginColumn,endLine,endColumn"
headers["Class"]="id,parent"
headers["Expression_location"]="id,location"
headers["Statement_location"]="id,location"
headers["Module"]="id"
headers["Expression"]="id,kind,parent,idx"
headers["Statement"]="id,kind,parent,idx"
headers["Variable"]="id,scope,name"
headers["VarAccess"]="id,parent"
headers["Function"]="id,parent"


for i in "${!headers[@]}"
do
    codeql query run -o "$dir"/"$i".bqrs -d $db "python-edb-queries/$i".ql
    codeql bqrs decode --output="$dir/$i".csv --format=csv --entities=id "$dir/$i".bqrs
    rm -rf "$dir/$i".bqrs
done
