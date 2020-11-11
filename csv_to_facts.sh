#!/bin/bash
mkdir -p codeql-"$1"
for i in $(ls "$1"); do cat "$1"/$i | tail -n +2 | ./csv2tab > codeql-"$1"/${i%.*}.facts; done; tput bel
