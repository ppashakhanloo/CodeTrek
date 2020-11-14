#!/usr/local/bin/bash

declare -A headers

headers["ClassObject"]="id,inferredType"
headers["Location"]="id,beginLine,beginColumn,endLine,endColumn"
headers["Class"]="id,parent"
headers["Expression_location"]="id,location"
headers["Statement_location"]="id,location"
headers["Module"]="id"
headers["Expression"]="id,kind,idx,module,parent"
headers["Statement"]="id,kind,parent"
headers["Variable"]="id,scope"
headers["Function"]="id,location,parent"


for i in "${!headers[@]}"
do
    codeql query run -o "$i".bqrs  -d pardis-database vscode-codeql-starter/codeql-custom-queries-python/"$i".ql
    codeql bqrs decode --output="$i".csv --format=csv --entities=id "$i".bqrs
    new_header="${headers[$i]}"
    sed -i.bk "1s/.*/$new_header/" "$i".csv
    rm "$i".csv.bk "$i".bqrs
done
