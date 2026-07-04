#!/usr/bin/env bash
# Block subagent fold-back if secrets appear in the staged diff or tests fail.
set -e

if git diff --cached | grep -iE "(api[_-]?key|secret|password|token)[[:space:]]*=" ; then
  echo "Potential secret in staged diff — blocking fold-back." >&2
  exit 1
fi

if [ -d "tests" ] && ls tests/*.py >/dev/null 2>&1; then
  python -m pytest -q || { echo "Tests failing — blocking fold-back." >&2; exit 1; }
fi

exit 0