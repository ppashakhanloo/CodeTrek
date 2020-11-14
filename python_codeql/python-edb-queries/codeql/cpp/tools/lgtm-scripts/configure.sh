#! /bin/bash
set -eu

if [ "${LGTM_CONFIGURE_COMMAND:-}" = "" ]; then
    LGTM_CONFIGURE_COMMAND="${CODEQL_EXTRACTOR_CPP_ROOT}/tools/do-prebuild"
fi

if [ -x /opt/deptrace/deptrace ]; then
  /opt/deptrace/deptrace \
    /opt/dist/tools/linux64/preload_tracer \
    "${LGTM_CONFIGURE_COMMAND}"
else
  "${LGTM_CONFIGURE_COMMAND}"
fi

