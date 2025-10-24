"""
sweep_hopper_gemm.py

Run the Hopper CuTe DSL dense_gemm example across a sweep of tunable params
(CTA tiles, cluster shapes, memory layouts) for a list of problem sizes from CSV.
Only successful + correct runs (unless --skip_ref_check) are recorded.

Defaults:
  --config        ./tune_config_hopper.yaml
  --out           ./sweep_results_hopper.csv

See tune_config_hopper.yaml to change the sweep space and problems CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


THIS_DIR = Path(__file__).parent
RUN_ONE = THIS_DIR / "run_one_hopper_config.py"
if not RUN_ONE.exists():
    raise FileNotFoundError(f"Missing helper script: {RUN_ONE}")


def parse_triplet_list(entries):
    out = []
    for item in entries:
        a = str(item).strip().lower().replace(" ", "")
        parts = a.split("x") if "x" in a else a.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad tile triplet '{item}' (want MxNxK)")
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def parse_pair_list(entries):
    out = []
    for item in entries:
        a = str(item).strip().lower().replace(" ", "")
        parts = a.split("x") if "x" in a else a.split(":")
        if len(parts) != 2:
            raise ValueError(f"Bad cluster shape '{item}' (want MxN)")
        out.append((int(parts[0]), int(parts[1])))
    return out


def parse_layout_triplets(entries):
    valid_a = set(["m", "k"])
    valid_b = set(["n", "k"])
    valid_c = set(["n", "m"])
    out = []
    for trip in entries:
        t = str(trip).strip().lower()
        if len(t) != 3:
            raise ValueError("Layout triplet must be 3 letters (e.g., kkn)")
        a, b, c = t[0], t[1], t[2]
        if a not in valid_a or b not in valid_b or c not in valid_c:
            raise ValueError(f"Invalid layout triplet '{t}'")
        out.append((a, b, c))
    return out


def gflops(M, N, K, us):
    if us <= 0:
        return float("nan")
    return (2.0 * M * N * K) / (us * 1e-6) / 1e9


def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    txt = path.read_text()
    if path.suffix in [".yml", ".yaml"]:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to read YAML configs. Install 'pyyaml' or provide a JSON config."
            )
        cfg = yaml.safe_load(txt)
    else:
        cfg = json.loads(txt)
    tiles = parse_triplet_list(cfg["tile_list"])
    clusters = parse_pair_list(cfg["cluster_list"])
    layouts = parse_layout_triplets(cfg["layouts"])
    iters = int(cfg.get("iters", 50))
    warmup = int(cfg.get("warmup", 5))
    use_cold_l2 = bool(cfg.get("use_cold_l2", False))
    tolerance = float(cfg.get("tolerance", 1e-2))
    timeout_sec = int(cfg.get("timeout_sec", 300))
    problems_csv = cfg.get("problems_csv", str(THIS_DIR / "problems_hopper.csv"))
    return {
        "tile_list": tiles,
        "cluster_list": clusters,
        "layouts": layouts,
        "iters": iters,
        "warmup": warmup,
        "use_cold_l2": use_cold_l2,
        "tolerance": tolerance,
        "timeout_sec": timeout_sec,
        "problems_csv": str(problems_csv),
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


FIELDNAMES = [
    "M", "N", "K", "L",
    "a_major", "b_major", "c_major",
    "tile_m", "tile_n", "tile_k",
    "cluster_m", "cluster_n",
    "avg_us", "gflops",
]


def ensure_csv_with_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()


def run_one(cfg_row, iters, warmup, tol, use_cold_l2, timeout_sec, skip_ref_check=False):
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
        "--tile", f"{cfg_row['tile_m']}x{cfg_row['tile_n']}x{cfg_row['tile_k']}",
        "--cluster", f"{cfg_row['cluster_m']}x{cfg_row['cluster_n']}",
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--tolerance", str(tol),
    ]
    if use_cold_l2:
        args.append("--use_cold_l2")
    if skip_ref_check:
        args.append("--skip_ref_check")

    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
        out = (p.stdout or "").strip()
        if out:
            try:
                return json.loads(out.splitlines()[-1])
            except json.JSONDecodeError:
                return {"ok": False, "kind": "fail", "error": f"bad-json: {out[-200:]} stderr: {(p.stderr or '')[-200:]}"}
        else:
            return {"ok": False, "kind": "fail", "error": f"empty-stdout rc={p.returncode} stderr: {(p.stderr or '')[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "kind": "fail", "error": f"timeout after {timeout_sec}s"}


def main():
    ap = argparse.ArgumentParser(description="Sweep Hopper GEMM over problems and tunables")
    ap.add_argument("--config", type=str, default=str(THIS_DIR / "tune_config_hopper.yaml"))
    ap.add_argument("--out", type=str, default=str(THIS_DIR / "sweep_results_hopper.csv"))
    ap.add_argument("--skip_ref_check", action="store_true")
    ap.add_argument("--timeout_sec", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    problems = load_problems(Path(cfg["problems_csv"]))
    out_csv = Path(args.out)
    ensure_csv_with_header(out_csv)

    tried = 0
    succeeded = 0

    for (M, N, K) in problems:
        for (a_major, b_major, c_major) in cfg["layouts"]:
            for (tm, tn, tk) in cfg["tile_list"]:
                for (cm, cn) in cfg["cluster_list"]:
                    tried += 1
                    row_cfg = {
                        "M": M, "N": N, "K": K, "L": 1,
                        "a_major": a_major, "b_major": b_major, "c_major": c_major,
                        "tile_m": tm, "tile_n": tn, "tile_k": tk,
                        "cluster_m": cm, "cluster_n": cn,
                    }
                    res = run_one(
                        row_cfg,
                        iters=cfg["iters"],
                        warmup=cfg["warmup"],
                        tol=cfg["tolerance"],
                        use_cold_l2=cfg["use_cold_l2"],
                        timeout_sec=(args.timeout_sec if args.timeout_sec is not None else cfg["timeout_sec"]),
                        skip_ref_check=args.skip_ref_check,
                    )
                    if res.get("ok"):
                        us = float(res["elapsed_us"])
                        succeeded += 1
                        with out_csv.open("a", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                            w.writerow({
                                **row_cfg,
                                "avg_us": us,
                                "gflops": gflops(M, N, K, us),
                            })
                            f.flush()
                        print(f"[OK] {row_cfg} -> {us:.2f} us, {gflops(M,N,K,us):.2f} GFLOPs")
                    else:
                        kind = res.get("kind", "fail")
                        err = res.get("error", "")
                        tag = "[skip]" if kind == "skip" else "[fail]"
                        print(f"{tag} {row_cfg} -> {err}")

    print(f"Completed {succeeded}/{tried} valid+checked configs. Results -> {out_csv}")


if __name__ == "__main__":
    main()
