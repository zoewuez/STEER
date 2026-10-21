"""
Merge scTM scores from a sweep of sampling runs into one CSV

Each run directory is expected to contain the outputs of sample.py
(sampled_angles/, sampler_config.json) and sctm.py (sctm_scores.json).

Usage:
    python merge_sctm.py RUNDIR [RUNDIR ...] -o combined_sctm.csv
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import *

import pandas as pd

from foldingdiff import sampling

GENERATED_RE = re.compile(r"generated_(\d+)")


def find_run_dirs(root: Path) -> List[Path]:
    """Return every directory under root that holds an sctm_scores.json"""
    return sorted(p.parent for p in root.rglob("sctm_scores.json"))


def collect_run(rundir: Path) -> pd.DataFrame:
    with open(rundir / "sctm_scores.json") as source:
        scores = json.load(source)
    config_path = rundir / "sampler_config.json"
    config = json.load(open(config_path)) if config_path.is_file() else {}
    schedule = config.get("replica_schedule")

    rows = []
    for name, score in scores.items():
        idx = int(GENERATED_RE.search(name).group(1))
        angles = rundir / "sampled_angles" / f"generated_{idx}.csv.gz"
        length = len(pd.read_csv(angles, index_col=0))
        n_replicas = (
            sampling.n_replicas_for_length(length, schedule)
            if schedule
            else config.get("n_replicas")
        )
        rows.append(
            {
                "run": str(rundir),
                "sampler": config.get("sampler"),
                "lam_start": config.get("lam_start"),
                "lam_end": config.get("lam_end"),
                "seed": config.get("seed"),
                "sample": idx,
                "length": length,
                "n_replicas": n_replicas,
                "sctm": score,
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dirs", type=str, nargs="+", help="Directories to search for runs")
    parser.add_argument(
        "-o", "--outfile", type=str, default="combined_sctm.csv", help="Output CSV"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = [d for root in args.dirs for d in find_run_dirs(Path(root))]
    if not run_dirs:
        raise SystemExit("No sctm_scores.json found")

    df = pd.concat([collect_run(d) for d in run_dirs], ignore_index=True)
    df = df.sort_values(["lam_start", "length", "run", "sample"])
    df.to_csv(args.outfile, index=False)
    logging.info(f"Wrote {len(df)} rows from {len(run_dirs)} runs to {args.outfile}")

    summary = df.groupby(["sampler", "lam_start"])["sctm"].agg(
        n="size", mean="mean", designable=lambda s: (s >= 0.5).mean()
    )
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
