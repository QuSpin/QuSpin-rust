#!/bin/bash

set -eo pipefail

clang_format="$1"
shift
outfile="$1"
shift
"$clang_format" "$@" 2>&1 | tee "$outfile" >&2
