#!/usr/bin/env bash
# Self-contained ClawTrace pipeline for one SWE-bench instance.
#
# Usage: ./run_swebench_oneshot.sh N
#   N = 1-indexed line number in swebench_lite.jsonl.
#
# Per-instance flow:
#   1. parse line N → instance_id, repo, base_commit
#   2. partial-clone the repo into ./swebench-repos/<instance_id>/
#   3. run profile_swebench.py against just this instance
#      → emits .jsonl, .diff, .predictions.json under traces/<episode>/
#   4. run analyzer.py and token_trace_gen.py on the produced trace
#      → adds .analysis.txt and .tokens.jsonl alongside
#   5. remove the clone (trap fires on EXIT/INT/TERM so killed runs don't leak)
#
# Sibling to run_swebench.sh, which assumes the repo is already cloned and
# leaves it in place.

set -uo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <line_number>" >&2
    exit 2
fi

n="$1"
if ! [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: N must be a positive integer; got '$n'" >&2
    exit 2
fi

instances_file="swebench_lite.jsonl"
if [ ! -f "$instances_file" ]; then
    echo "Error: $instances_file not found" >&2
    exit 1
fi

line=$(sed -n "${n}p" "$instances_file")
if [ -z "$line" ]; then
    echo "Error: $instances_file has fewer than $n lines" >&2
    exit 1
fi

# Parse via Python — safer than shell text manipulation on JSON.
parsed=$(echo "$line" | python -c "
import sys, json
o = json.loads(sys.stdin.read())
print(o['instance_id'], o['repo'], o['base_commit'])
")
read -r instance_id repo base_commit <<< "$parsed"
if [ -z "${instance_id:-}" ] || [ -z "${repo:-}" ] || [ -z "${base_commit:-}" ]; then
    echo "Error: failed to parse line $n as a SWE-bench instance" >&2
    exit 1
fi

repo_url="https://github.com/${repo}.git"
repo_path="./swebench-repos/${instance_id}"
output_file=$(mktemp)

cleanup() {
    rm -f "$output_file"
    if [ -d "$repo_path" ]; then
        echo "Cleanup: removing $repo_path" >&2
        rm -rf "$repo_path"
    fi
}
trap cleanup EXIT INT TERM

if [ -d "$repo_path" ]; then
    echo "Note: $repo_path already exists; removing for fresh clone"
    rm -rf "$repo_path"
fi

echo "============================================================"
echo "Instance:    $instance_id (line $n)"
echo "Repo:        $repo_url"
echo "Base commit: $base_commit"
echo "============================================================"

# Partial clone: full commit graph, no blobs. Blobs stream on demand when
# the dispenser's reset_repo does `git reset --hard <base_commit>`. ~10x
# faster than a full clone for typical SWE-bench repos.
echo "==> Cloning (partial; blobs stream on demand)"
if ! git clone --filter=blob:none "$repo_url" "$repo_path"; then
    echo "Error: clone failed" >&2
    exit 1
fi

if ! git -C "$repo_path" cat-file -e "${base_commit}^{commit}" 2>/dev/null; then
    echo "Error: base_commit $base_commit not in cloned repo" >&2
    exit 1
fi

echo "==> Running profile_swebench.py"
python profile_swebench.py \
    --instances "$instances_file" \
    --repos-dir ./swebench-repos \
    --instance-id "$instance_id" \
    2>&1 | tee "$output_file"
rc=${PIPESTATUS[0]}

trace=$(grep -oE 'trace=[^ ]+' "$output_file" | head -n1 | sed 's/^trace=//')
if [ -n "$trace" ] && [ -f "$trace" ]; then
    echo
    echo "==> Analyzing $trace"
    python analyzer.py "$trace"
    echo
    echo "==> Generating token trace for $trace"
    python token_trace_gen.py "$trace"
else
    echo "Warning: no trace path found in profile_swebench.py output" >&2
fi

# trap fires on exit and removes the clone.
exit "$rc"
