#!/bin/bash

rm -rf codeql_dir
mkdir codeql_dir
cd codeql_dir
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.4.1/codeql-linux64.zip
unzip codeql-linux64.zip
git clone https://github.com/github/codeql.git codeql_repo

echo -e "\nPATH=$PWD/codeql:\$PATH" >> ~/.bashrc
source ~/.bashrc
