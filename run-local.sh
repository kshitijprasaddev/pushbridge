#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m pushbridge "$@"
elif [[ -x ../../work/venv/bin/python ]]; then
  exec ../../work/venv/bin/python -m pushbridge "$@"
else
  exec python3 -m pushbridge "$@"
fi
