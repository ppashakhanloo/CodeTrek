#!/bin/sh

set -eu

# Directory where the autobuild scripts live.
AUTOBUILD_ROOT="$CODEQL_EXTRACTOR_CPP_ROOT/tools"

"$AUTOBUILD_ROOT/do-prebuild"

"$AUTOBUILD_ROOT/do-build"
