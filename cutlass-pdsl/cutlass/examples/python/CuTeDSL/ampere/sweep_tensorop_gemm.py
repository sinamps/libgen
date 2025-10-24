
"""
sweep_tensorop_gemm.py

Runs the tunable CuTe DSL tensorcore GEMM example across a sweep of tunable params
(CTA tiles, stages, atom layouts, memory layouts) for a LIST of problem sizes loaded
from a CSV file. Only records rows that pass correctness checking.

Defaults:
  --problems_csv ./problems.csv
  --config       ./tune_config.yaml
  --out          ./sweep_results.csv

Edit tune_config.yaml to change sweep parameters.
"""

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# third-party: pyyaml is commonly available in research envs; if not, fallback to json
try:
    import yaml  # type: ignore
except Exception:
    yaml = None
import json

# We will run each configuration in an isolated subprocess via run_one_config.py
THIS_DIR = Path(__file__).parent
RUN_ONE = THIS_DIR / "run_one_config.py"
if not RUN_ONE.exists():
    raise FileNotFoundError(f"Missing helper script: {RUN_ONE}")


def parse_cta_list(entries):
    out = []
    for item in entries:
        a = str(item).strip().lower().replace(" ", "")
        if "x" in a:
            ms = [int(z) for z in a.split("x")]
        else:
            ms = [int(z) for z in a.split(":")]
        if len(ms) != 3:
            raise ValueError(f"Bad CTA shape '{item}' (want MxNxK)")
        out.append(tuple(ms))
    return out


def parse_atom_layouts(entries):
    out = []
    for item in entries:
        a = str(item).strip().lower().replace(" ", "")
        if "x" in a:
            ms = [int(z) for z in a.split("x")]
        else:
            ms = [int(z) for z in a.split(":")]
        if len(ms) != 3:
            raise ValueError(f"Bad atom layout '{item}' (want MxNxK atoms)")
        out.append(tuple(ms))
    return out


def parse_layout_triplets(entries):
    # triplets like "mnn", "kmn"
    valid_a = set(["m", "k"])
    valid_b = set(["n", "k"])
    valid_c = set(["n", "m"])
    out = []
    for trip in entries:
        t = str(trip).strip().lower()
        if len(t) != 3:
            raise ValueError("Layout triplet must be 3 letters (e.g., mnn)")
        a, b, c = t[0], t[1], t[2]
        if a not in valid_a or b not in valid_b or c not in valid_c:
            raise ValueError(f"Invalid layout triplet '{t}'")
        out.append((a, b, c))
    return out


def gflops(M, N, K, us):
    if us <= 0:
        return float("nan")
    ops = 2 * M * N * K
    return (ops / (us * 1e-6)) / 1e9


FIELDNAMES = [
    "M","N","K","L",
    "a_major","b_major","c_major",
    "cta_m","cta_n","cta_k","stages",
    "atom_m","atom_n","atom_k",
    "avg_us","gflops",
]


def ensure_csv_with_header(path: Path, fieldnames):
    """Create CSV and write header if the file doesn't exist or is empty.
    If exists and non-empty, assume header already present and append rows later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    txt = path.read_text()
    if yaml is not None and (path.suffix in [".yml", ".yaml"]):
        cfg = yaml.safe_load(txt)
    else:
        cfg = json.loads(txt)
    # normalize fields
    ctas = parse_cta_list(cfg["cta_list"])
    stages = [int(s) for s in cfg["stages"]]
    atoms = parse_atom_layouts(cfg["atom_layouts"])
    layouts = parse_layout_triplets(cfg["layouts"])
    iters = int(cfg.get("iters", 50))
    warmup = int(cfg.get("warmup", 5))
    use_cold_l2 = bool(cfg.get("use_cold_l2", False))
    timeout_sec = int(cfg.get("timeout_sec", 300))  # per-config timeout (seconds)
    return {
        "cta_list": ctas,
        "stages": stages,
        "atom_layouts": atoms,
        "layouts": layouts,
        "iters": iters,
        "warmup": warmup,
        "use_cold_l2": use_cold_l2,
        "timeout_sec": timeout_sec,
    }


def load_problems(csv_path: Path):
    probs = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["m"])
            n = int(row["n"])
            k = int(row["k"])
            probs.append((m, n, k))
    return probs


def run_one_subprocess(cfg_row, iters, warmup, use_cold_l2, timeout_sec, skip_ref_check=False):
    """Invoke run_one_config.py with the given config and return parsed JSON or raise on protocol error."""
    args = [
        sys.executable,
        str(RUN_ONE),
        "--M", str(cfg_row["M"]),
        "--N", str(cfg_row["N"]),
        "--K", str(cfg_row["K"]),
        "--L", str(cfg_row["L"]),
        "--a_major", cfg_row["a_major"],
        "--b_major", cfg_row["b_major"],
        "--c_major", cfg_row["c_major"],
        "--cta_m", str(cfg_row["cta_m"]),
        "--cta_n", str(cfg_row["cta_n"]),
        "--cta_k", str(cfg_row["cta_k"]),
        "--stages", str(cfg_row["stages"]),
        "--atom_m", str(cfg_row["atom_m"]),
        "--atom_n", str(cfg_row["atom_n"]),
        "--atom_k", str(cfg_row["atom_k"]),
        "--iters", str(iters),
        "--warmup", str(warmup),
    ]
    if use_cold_l2:
        args.append("--use_cold_l2")
    if skip_ref_check:
        args.append("--skip_ref_check")

    # Run and capture a single-line JSON response
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        # Ensure process is terminated; subprocess.run already kills on timeout, but be defensive.
        return {"ok": False, "kind": "fail", "error": f"timeout after {timeout_sec}s"}

    # Prefer stdout JSON if available; if empty, synthesize a fail
    if stdout:
        try:
            return json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            # Include last 200 chars of stderr for context
            return {"ok": False, "kind": "fail", "error": f"bad-json: {stdout[-200:]} stderr: {stderr[-200:]}"}
    else:
        return {"ok": False, "kind": "fail", "error": f"empty-stdout rc={proc.returncode} stderr: {stderr[-200:]}"}


def main():
    p = argparse.ArgumentParser(description="Sweep tuning params for CuTe tensorop GEMM over many problems")
    p.add_argument("--problems_csv", type=str, default=str(Path(__file__).with_name("problems.csv")))
    p.add_argument("--config", type=str, default=str(Path(__file__).with_name("tune_config.yaml")))
    p.add_argument("--out", type=str, default=str(Path(__file__).with_name("sweep_results.csv")))
    # Expert override (optional): allow user to disable correctness check (not default)
    p.add_argument("--skip_ref_check", action="store_true", help="Not recommended: record even without checking reference")
    # Optional per-config timeout override
    p.add_argument("--timeout_sec", type=int, default=None, help="Per-config timeout in seconds (defaults to tune_config.yaml or 300)")
    args = p.parse_args()

    problems_csv = Path(args.problems_csv)
    cfg_path = Path(args.config)
    out_csv = Path(args.out)

    cfg = load_config(cfg_path)
    problems = load_problems(problems_csv)

    # Prepare output CSV and write header once
    ensure_csv_with_header(out_csv, FIELDNAMES)

    tried = 0
    succeeded = 0

    for (M, N, K) in problems:
        for (a_major, b_major, c_major) in cfg["layouts"]:
            for cta in cfg["cta_list"]:
                for stages in cfg["stages"]:
                    for atoms in cfg["atom_layouts"]:
                        tried += 1
                        cfg_row = {
                            "M": M, "N": N, "K": K, "L": 1,
                            "a_major": a_major, "b_major": b_major, "c_major": c_major,
                            "cta_m": cta[0], "cta_n": cta[1], "cta_k": cta[2],
                            "stages": stages,
                            "atom_m": atoms[0], "atom_n": atoms[1], "atom_k": atoms[2],
                        }
                        # Execute in isolated subprocess; do not pre-filter, try them all.
                        res = run_one_subprocess(
                            cfg_row,
                            iters=cfg["iters"],
                            warmup=cfg["warmup"],
                            use_cold_l2=cfg["use_cold_l2"],
                            timeout_sec=(args.timeout_sec if args.timeout_sec is not None else cfg["timeout_sec"]),
                            skip_ref_check=args.skip_ref_check,
                        )
                        if res.get("ok"):
                            elapsed = float(res["elapsed_us"])
                            succeeded += 1
                            row_out = {
                                **cfg_row,
                                "avg_us": elapsed,
                                "gflops": gflops(M, N, K, elapsed),
                            }
                            # Append to CSV immediately for real-time progress monitoring
                            with out_csv.open("a", newline="") as f:
                                w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                                w.writerow({k: row_out.get(k, "") for k in FIELDNAMES})
                                f.flush()
                            print(f"[OK] {cfg_row} -> {elapsed:.2f} us, {gflops(M,N,K,elapsed):.2f} GFLOPs")
                        else:
                            kind = res.get("kind", "fail")
                            err = res.get("error", "")
                            tag = "[skip]" if kind == "skip" else "[fail]"
                            print(f"{tag} {cfg_row} -> {err}")

    print(f"Completed {succeeded}/{tried} valid+checked configs. Results -> {out_csv}")


if __name__ == "__main__":
    main()
