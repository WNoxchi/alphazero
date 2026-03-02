#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <N>"
  echo "  N = number of codex iterations to run"
  exit 1
fi

N=$1
BUILD_PROMPT="$(cat prompts/PROMPT_build.md)"$'\n\nReturn \"TASK COMPLETE\" and HALT when done.'
REVIEW_PROMPT="$(cat prompts/PROMPT_review.md)"$'\n\nReturn \"REVIEW COMPLETE\" and HALT when done.'
BUILD_OUTFILE="logs/build_output.txt"
REVIEW_OUTFILE="logs/review_output.json"
mkdir -p logs

for i in $(seq 1 "$N"); do
  echo "=== Iteration $i / $N — BUILD ==="

  codex exec \
    --color always \
    --full-auto \
    -o "$BUILD_OUTFILE" \
    "$BUILD_PROMPT"

  if grep -q "TASK COMPLETE" "$BUILD_OUTFILE" 2>/dev/null; then
    echo ">>> Builder returned TASK COMPLETE on iteration $i"
  else
    echo ">>> WARNING: TASK COMPLETE not found in builder output on iteration $i"
  fi

  echo "=== Iteration $i / $N — REVIEW ==="

  claude -p "$REVIEW_PROMPT" \
    --allowedTools "Read Write Edit Glob Grep Bash(git:*)" \
    --output-format json \
    --verbose \
    2>&1 | tee /dev/stderr | jq '.' > "$REVIEW_OUTFILE"

  if jq -r '.result // .content // empty' "$REVIEW_OUTFILE" 2>/dev/null | grep -q "REVIEW COMPLETE"; then
    echo ">>> Reviewer returned REVIEW COMPLETE on iteration $i"
  else
    echo ">>> WARNING: REVIEW COMPLETE not found in reviewer output on iteration $i"
  fi
done

echo "=== All $N iterations finished ==="
