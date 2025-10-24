"""
summarize_max_gflops.py

Reads a sweep results CSV (produced by sweep_tensorop_gemm.py) and writes a
compact CSV with the maximum average GFLOPs per problem size (M,N,K,L),
including the configuration that achieved it.

Defaults:
  --in  ./sweep_results.csv
  --out ./best_by_problem.csv
    --filter_layout mnn   # filter rows by (a_major,b_major,c_major); use 'all' to disable
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def to_int(v):
    try:
        return int(v)
    except Exception:
        return int(float(v))


def to_float(v):
    return float(v)


def main():
    p = argparse.ArgumentParser(description="Summarize max GFLOPs per problem size from sweep results")
    p.add_argument("--in", dest="inp", type=str, default=str(Path(__file__).with_name("sweep_results.csv")))
    p.add_argument("--out", dest="out", type=str, default=str(Path(__file__).with_name("best_by_problem.csv")))
    p.add_argument(
        "--filter_layout",
        type=str,
        default="mnn",
        help="Filter by layout triplet (a_major,b_major,c_major). Example: mnn, kmn. Use 'all' to disable.",
    )
    args = p.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)

    if not inp.exists():
        raise FileNotFoundError(f"Input CSV not found: {inp}")

    best = {}  # key: (M,N,K,L) -> row dict of best config

    # Parse filter
    flt = (args.filter_layout or "").strip().lower()
    if flt and flt != "all":
        if len(flt) != 3:
            raise ValueError("--filter_layout must be 3 chars (e.g., mnn) or 'all'")
        fa, fb, fc = flt[0], flt[1], flt[2]
    else:
        fa = fb = fc = None  # no filtering

    with inp.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                M = to_int(row["M"])  # required
                N = to_int(row["N"])  # required
                K = to_int(row["K"])  # required
                L = to_int(row.get("L", 1))
                gfl = to_float(row["gflops"])  # required
                us = to_float(row.get("avg_us", "nan"))
            except Exception:
                # skip malformed rows
                continue

            # Apply optional layout filter
            if fa is not None:
                a_val = str(row.get("a_major", "")).strip().lower()
                b_val = str(row.get("b_major", "")).strip().lower()
                c_val = str(row.get("c_major", "")).strip().lower()
                if not (a_val == fa and b_val == fb and c_val == fc):
                    continue

            key = (M, N, K, L)
            prev = best.get(key)
            if prev is None:
                best[key] = {
                    **row,
                    "M": M,
                    "N": N,
                    "K": K,
                    "L": L,
                    "gflops": gfl,
                    "avg_us": us,
                }
            else:
                prev_g = float(prev.get("gflops", 0.0))
                # prefer higher GFLOPs; on tie, pick lower avg_us if available
                if (gfl > prev_g) or (gfl == prev_g and us < float(prev.get("avg_us", float("inf")))):
                    best[key] = {
                        **row,
                        "M": M,
                        "N": N,
                        "K": K,
                        "L": L,
                        "gflops": gfl,
                        "avg_us": us,
                    }

    # Output fields: problem + best score + config that achieved it
    fieldnames = [
        "M", "N", "K", "L",
        "max_gflops", "avg_us",
        "a_major", "b_major", "c_major",
        "cta_m", "cta_n", "cta_k", "stages",
        "atom_m", "atom_n", "atom_k",
    ]

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # sort by (M,N,K,L) for stable output
        for (M, N, K, L) in sorted(best.keys()):
            r = best[(M, N, K, L)]
            w.writerow({
                "M": M,
                "N": N,
                "K": K,
                "L": L,
                "max_gflops": r.get("gflops", ""),
                "avg_us": r.get("avg_us", ""),
                "a_major": r.get("a_major", ""),
                "b_major": r.get("b_major", ""),
                "c_major": r.get("c_major", ""),
                "cta_m": r.get("cta_m", ""),
                "cta_n": r.get("cta_n", ""),
                "cta_k": r.get("cta_k", ""),
                "stages": r.get("stages", ""),
                "atom_m": r.get("atom_m", ""),
                "atom_n": r.get("atom_n", ""),
                "atom_k": r.get("atom_k", ""),
            })


if __name__ == "__main__":
    main()
