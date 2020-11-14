#! /bin/bash
set -eu

TOOLS_DIR="${CODEQL_EXTRACTOR_CPP_ROOT}/tools"

if [ "${LGTM_INDEX_BUILD_COMMAND:-}" = "" ]; then
  LGTM_INDEX_BUILD_COMMAND="${TOOLS_DIR}/do-build"
fi

if [ -x /opt/deptrace/deptrace ]; then
  /opt/deptrace/deptrace \
    "${CODEQL_DIST}/codeql" database trace-command "${CODEQL_EXTRACTOR_CPP_WIP_DATABASE}" -- \
    "${LGTM_INDEX_BUILD_COMMAND}"
else
    "${CODEQL_DIST}/codeql" database trace-command "${CODEQL_EXTRACTOR_CPP_WIP_DATABASE}" -- \
    "${LGTM_INDEX_BUILD_COMMAND}"
fi

# Produce a trap file to capture the mapping from headers to packages.
"${TOOLS_DIR}/lgtm-scripts/header_packages.py"
