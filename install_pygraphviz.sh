#!/usr/bin/env bash

wget https://www2.graphviz.org/Packages/stable/portable_source/graphviz-2.44.1.tar.gz && tar -xzf graphviz-2.44.1.tar.gz && mv graphviz-2.44.1 graphviz

cd graphviz
path="$(pwd)"
./configure --prefix="$path/graphviz_bins"
make
make install

export PKG_CONFIG_PATH="$path/graphviz_bins/lib/pkgconfig/"
pip install --global-option=build_ext --global-option="-I$path/graphviz_bins/include" --global-option="-L$path/graphviz_bins/lib" pygraphviz
