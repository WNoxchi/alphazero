#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <N>"
  echo "  N = number of codex iterations to run"
  exit 1
fi

N=$1
PROMPT='Read and execute PROMPT_build.md. Emit "TASK COMPLETE" and HALT when done.'
OUTFILE=$(mktemp)

for i in $(seq 1 "$N"); do
  echo "=== Iteration $i / $N ==="

  codex exec \
    --color always \
    -o "$OUTFILE" \
    "$PROMPT"

  if grep -q "TASK COMPLETE" "$OUTFILE" 2>/dev/null; then
    echo ">>> Codex emitted TASK COMPLETE on iteration $i"
  else
    echo ">>> WARNING: TASK COMPLETE not found in output on iteration $i"
  fi
done

rm -f "$OUTFILE"
echo "=== All $N iterations finished ==="
