#!/bin/sh

set -eu

exec "${CODEQL_JAVA_HOME}/bin/java" \
    -jar "$CODEQL_EXTRACTOR_XML_ROOT/tools/xml-extractor.jar" \
        --fileList="$1" \
        --sourceArchiveDir="$CODEQL_EXTRACTOR_XML_SOURCE_ARCHIVE_DIR" \
        --outputDir="$CODEQL_EXTRACTOR_XML_TRAP_DIR"
