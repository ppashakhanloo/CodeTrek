#!/bin/sh

set -eu

exec "${CODEQL_DIST}/codeql" database index-files \
    --include-extension=.xml \
    --size-limit=5m \
    --language=xml \
    "$CODEQL_EXTRACTOR_XML_WIP_DATABASE"