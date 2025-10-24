"""
run_one_hopper_config.py

Helper to execute a single Hopper GEMM configuration (dense_gemm.py) in an isolated
Python process. Emits one JSON line on stdout:

  {"ok": true,  "elapsed_us": <float>}        # success
  {"ok": false, "kind": "skip", "error": <str>}  # AssertionError / constraints / correctness
  {"ok": false, "kind": "fail", "error": <str>}  # other exceptions
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path


def parse_triplet_mnk(s: str) -> tuple[int, int, int]:
    a = str(s).strip().lower().replace(" ", "")
    parts = a.split("x") if "x" in a else a.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected MxNxK for tile shape")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def parse_pair_mn(s: str) -> tuple[int, int]:
    a = str(s).strip().lower().replace(" ", "")
    parts = a.split("x") if "x" in a else a.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected MxN for cluster shape")
    return (int(parts[0]), int(parts[1]))


def main():
    p = argparse.ArgumentParser(description="Run a single Hopper GEMM config and return JSON result")
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--L", type=int, default=1)
    p.add_argument("--a_major", choices=["m", "k"], required=True)
    p.add_argument("--b_major", choices=["n", "k"], required=True)
    p.add_argument("--c_major", choices=["n", "m"], required=True)
    p.add_argument("--tile", type=parse_triplet_mnk, required=True, help="Tile shape MxNxK")
    p.add_argument("--cluster", type=parse_pair_mn, required=True, help="Cluster shape MxN")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=1e-2)
    p.add_argument("--use_cold_l2", action="store_true")
    p.add_argument("--skip_ref_check", action="store_true")
    args = p.parse_args()

    # import target module locally
    this_dir = Path(__file__).parent
    sys.path.insert(0, str(this_dir))
    import dense_gemm as gemm_mod  # type: ignore

    mnkl = (args.M, args.N, args.K, args.L)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            elapsed = gemm_mod.run(
                mnkl=mnkl,
                a_dtype=gemm_mod.cutlass.Float16,
                b_dtype=gemm_mod.cutlass.Float16,
                c_dtype=gemm_mod.cutlass.Float16,
                acc_dtype=gemm_mod.cutlass.Float32,
                a_major=args.a_major,
                b_major=args.b_major,
                c_major=args.c_major,
                tile_shape_mnk=args.tile,
                cluster_shape_mn=args.cluster,
                tolerance=args.tolerance,
                warmup_iterations=args.warmup,
                iterations=args.iters,
                skip_ref_check=args.skip_ref_check,
                use_cold_l2=args.use_cold_l2,
            )
        print(json.dumps({"ok": True, "elapsed_us": float(elapsed)}))
    except AssertionError as e:
        print(json.dumps({"ok": False, "kind": "skip", "error": str(e)}))
    except Exception as e:
        print(json.dumps({"ok": False, "kind": "fail", "error": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
    main()
