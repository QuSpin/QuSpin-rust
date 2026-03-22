#!/bin/bash

set -eo pipefail

clang_tidy="$1"
shift
outfile="$1"
shift
"$clang_tidy" "$@" 2>&1 | tee "$outfile" | sed '/[0-9]\+ warnings\{0,1\} generated\./d' >&2
