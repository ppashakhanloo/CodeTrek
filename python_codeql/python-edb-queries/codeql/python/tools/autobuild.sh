#!/bin/sh

set -eu

# Legacy environment variables for the autobuild infrastructure.
LGTM_SRC="$(pwd)"
LGTM_WORKSPACE="$CODEQL_EXTRACTOR_PYTHON_SCRATCH_DIR"
export LGTM_SRC
export LGTM_WORKSPACE

if ! which python >/dev/null; then
    echo "ERROR: 'python' not found, it should be available when running 'which python' in your shell"
    exit 1
fi

exec python "$CODEQL_EXTRACTOR_PYTHON_ROOT/tools/index.py"
