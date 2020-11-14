#!/bin/sh

set -eu

exec "${CODEQL_DIST}/codeql" database index-files \
    --include-extension=.csv \
    --size-limit=5m \
    --language=csv \
    "$CODEQL_EXTRACTOR_CSV_WIP_DATABASE"
