#!/usr/bin/env bash
# End-to-end ClawTrace pipeline: SWE-bench dispenser → per-trace analysis.
# Forwards all args to profile_swebench.py.
#
# Examples:
#   ./run_swebench.sh --instance-id astropy__astropy-12907
#   ./run_swebench.sh --repo django/django --limit 3 --config '{"max_steps":30}'
#   ./run_swebench.sh --limit 5
#
# Each trace produced by the dispenser is analyzed in turn; the analyzer
# auto-saves its output as <trace_dir>/<trace_stem>.analysis.txt alongside
# the .jsonl + .diff + .predictions.json artifacts.

set -uo pipefail

cd "$(dirname "$0")"
if [ -z "${VIRTUAL_ENV:-}" ]; then
    for cand in .venv ../venv-clawtrace; do
        if [ -f "$cand/bin/activate" ]; then
            # shellcheck disable=SC1090
            source "$cand/bin/activate"
            break
        fi
    done
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        echo "Warning: no venv activated and no .venv/ or ../venv-clawtrace/ found; using \$(which python)" >&2
    fi
fi

output_file=$(mktemp)
trap 'rm -f "$output_file"' EXIT

python profile_swebench.py \
    --instances swebench_lite.jsonl \
    --repos-dir ./swebench-repos \
    "$@" \
    2>&1 | tee "$output_file"
dispenser_rc=${PIPESTATUS[0]}

grep -oE 'trace=[^ ]+' "$output_file" \
    | sed 's/^trace=//' \
    | while IFS= read -r trace; do
        if [ -f "$trace" ]; then
            echo
            echo "==> Analyzing $trace"
            python analyzer.py "$trace"
            echo "==> Generating token trace for $trace"
            python token_trace_gen.py "$trace"
        fi
    done

exit "$dispenser_rc"
