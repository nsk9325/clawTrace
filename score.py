"""Aggregate ``local_eval`` reports into a SWE-bench-style resolved rate.

Walks every ``eval-logs/<run_id>/report.json`` (or a user-supplied glob),
collects per-instance results, dedupes by instance_id (most-recent wins),
and prints both:

  - **Score among evaluated** — resolved / total of the instances you've
    actually run locally. Useful for iterating on agent correctness.
  - **Score on the full subset** — resolved / |subset|. Comparable to the
    SWE-bench leaderboard's headline number, but only meaningful once
    you've evaluated a large fraction of the subset.

Usage::

    python score.py                          # default: eval-logs/*/report.json on swe-bench_lite
    python score.py --subset swe-bench_lite  # explicit subset
    python score.py --reports 'eval-logs/*/report.json'  # custom glob
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


_SUBSET_SIZES = {
    "swe-bench_lite": 300,
    "swe-bench_verified": 500,
    "swe-bench-m": 245,
}


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _collect_latest_per_instance(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Most-recent-wins dedup: report files are walked in mtime order."""
    latest: dict[str, dict[str, Any]] = {}
    for _, report in reports:
        for inst in report.get("instances", []):
            latest[inst["instance_id"]] = inst
    return latest


def _format_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num / denom * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--reports",
        default="eval-logs/*/report.json",
        help="Glob for report.json files (default: eval-logs/*/report.json)",
    )
    p.add_argument(
        "--subset",
        default="swe-bench_lite",
        choices=sorted(_SUBSET_SIZES),
        help="SWE-bench subset for full-subset denominator (default: swe-bench_lite)",
    )
    p.add_argument(
        "--by-repo",
        action="store_true",
        help="Break down resolved rate per repo",
    )
    args = p.parse_args(argv)

    paths = sorted(Path(p) for p in glob.glob(args.reports))
    paths.sort(key=lambda x: x.stat().st_mtime)
    if not paths:
        print(f"No reports matched {args.reports!r}")
        return 1

    reports = [(p, _load_report(p)) for p in paths]
    latest = _collect_latest_per_instance(reports)

    total = len(latest)
    resolved = sum(1 for r in latest.values() if r["resolved"])
    errored = sum(1 for r in latest.values() if r.get("error"))
    subset_size = _SUBSET_SIZES[args.subset]

    print(f"Aggregating {len(paths)} report file(s); {total} unique instance(s) evaluated.")
    print(f"  Resolved among evaluated: {resolved}/{total}  ({_format_pct(resolved, total)})")
    print(f"  Resolved on {args.subset}:   {resolved}/{subset_size}  ({_format_pct(resolved, subset_size)})")
    if errored:
        print(f"  Errored (eval setup failed): {errored}")

    if args.by_repo:
        per_repo: dict[str, dict[str, int]] = defaultdict(lambda: {"resolved": 0, "total": 0})
        for inst_id, r in latest.items():
            # Convention from SWE-bench: instance_id is `<owner>__<repo>-<num>`
            repo = inst_id.rsplit("-", 1)[0].replace("__", "/")
            per_repo[repo]["total"] += 1
            if r["resolved"]:
                per_repo[repo]["resolved"] += 1
        print()
        print("Per repo:")
        for repo in sorted(per_repo):
            cnt = per_repo[repo]
            print(f"  {repo:<32} {cnt['resolved']}/{cnt['total']}  ({_format_pct(cnt['resolved'], cnt['total'])})")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
