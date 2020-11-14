#!/bin/bash

set -eu

"${CODEQL_JAVA_HOME}/bin/java" \
    -jar "$CODEQL_EXTRACTOR_JAVA_ROOT/tools/autobuild-fat.jar" \
    autoBuild --no-indexing || exit $?
