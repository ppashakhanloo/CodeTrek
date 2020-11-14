#!/bin/bash

set -eu
if [ "${LGTM_INDEX_XML_MODE:-default}" == "default" ]; then
"$CODEQL_DIST/codeql" database index-files \
    --include "**/AndroidManifest.xml" \
    --include "**/pom.xml" \
    --include "**/web.xml" \
    --size-limit 10m \
    --language xml \
    -- \
    "$CODEQL_EXTRACTOR_JAVA_WIP_DATABASE"
elif [ "${LGTM_INDEX_XML_MODE}" == "smart" ]; then
export CODEQL_EXTRACTOR_XML_PRIMARY_TAGS="faceted-project project plugin idea-plugin beans struts web-app module ui:UiBinder persistence"
"$CODEQL_DIST/codeql" database index-files \
    --include-extension=.xml \
    --size-limit 10m \
    --language xml \
    -- \
    "$CODEQL_EXTRACTOR_JAVA_WIP_DATABASE"
elif [ "${LGTM_INDEX_XML_MODE}" == "all" ]; then
"$CODEQL_DIST/codeql" database index-files \
    --include-extension=.xml \
    --size-limit 10m \
    --language xml \
    -- \
    "$CODEQL_EXTRACTOR_JAVA_WIP_DATABASE"
fi

if [ "${LGTM_INDEX_PROPERTIES_FILES:-false}" == "true" ]; then
"$CODEQL_DIST/codeql" database index-files \
    --include-extension=.properties \
    --size-limit=5m \
    --language properties \
    -- \
    "$CODEQL_EXTRACTOR_JAVA_WIP_DATABASE"
fi
