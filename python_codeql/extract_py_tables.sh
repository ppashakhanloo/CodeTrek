#!/usr/local/bin/bash

dir="$1"
db="$2"

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
    codeql query run -o "$dir"/"$i".bqrs -d $db "/Users/pardis/Google Drive/[University]/Sem 7/Research/CodeQL/vscode-codeql-starter/codeql-custom-queries-python/$i".ql
    codeql bqrs decode --output="$dir/$i".csv --format=csv --entities=id "$dir/$i".bqrs
    rm -rf "$dir/$i".bqrs
done
