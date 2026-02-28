#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <N>"
  echo "  N = number of codex iterations to run"
  exit 1
fi

N=$1
BUILD_PROMPT='Read and execute `prompts/PROMPT_build.md`. Return "TASK COMPLETE" and HALT when done.'
REVIEW_PROMPT='Read and execute `prompts/PROMPT_review.md`. Return "REVIEW COMPLETE" and HALT when done.'
OUTFILE=$(mktemp)

for i in $(seq 1 "$N"); do
  echo "=== Iteration $i / $N — BUILD ==="

  codex exec \
    --color always \
    -o "$OUTFILE" \
    "$BUILD_PROMPT"

  if grep -q "TASK COMPLETE" "$OUTFILE" 2>/dev/null; then
    echo ">>> Builder returned TASK COMPLETE on iteration $i"
  else
    echo ">>> WARNING: TASK COMPLETE not found in builder output on iteration $i"
  fi

  echo "=== Iteration $i / $N — REVIEW ==="

  codex exec \
    --color always \
    -o "$OUTFILE" \
    "$REVIEW_PROMPT"

  if grep -q "REVIEW COMPLETE" "$OUTFILE" 2>/dev/null; then
    echo ">>> Reviewer returned REVIEW COMPLETE on iteration $i"
  else
    echo ">>> WARNING: REVIEW COMPLETE not found in reviewer output on iteration $i"
  fi
done

rm -f "$OUTFILE"
echo "=== All $N iterations finished ==="
