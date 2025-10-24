"""
run_one_config.py

Helper to execute a single configuration of the tunable CuTe DSL tensorcore GEMM
in an isolated Python process. It suppresses verbose prints from the underlying
module and only emits a single JSON line to stdout with the result:

  {"ok": true,  "elapsed_us": <float>}        # success
  {"ok": false, "kind": "skip", "error": <str>}  # AssertionError (e.g., correctness/constraints)
  {"ok": false, "kind": "fail", "error": <str>}  # any other exception (compile/runtime)

This allows a parent sweep process to robustly continue even if the CUDA context
is corrupted by a bad config (e.g., illegal memory access), since each run is isolated.
"""

from __future__ import annotations

import argparse
import json
import io
import sys
import contextlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Run a single CuTe GEMM config and return JSON result")
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--L", type=int, default=1)
    p.add_argument("--a_major", choices=["m", "k"], required=True)
    p.add_argument("--b_major", choices=["n", "k"], required=True)
    p.add_argument("--c_major", choices=["n", "m"], required=True)
    p.add_argument("--cta_m", type=int, required=True)
    p.add_argument("--cta_n", type=int, required=True)
    p.add_argument("--cta_k", type=int, required=True)
    p.add_argument("--stages", type=int, required=True)
    p.add_argument("--atom_m", type=int, required=True)
    p.add_argument("--atom_n", type=int, required=True)
    p.add_argument("--atom_k", type=int, required=True)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--use_cold_l2", action="store_true")
    p.add_argument("--skip_ref_check", action="store_true")
    args = p.parse_args()

    # Ensure local import works
    this_dir = Path(__file__).parent
    sys.path.insert(0, str(this_dir))
    import tensorop_gemm_tunable as gemm_mod  # type: ignore

    # Build inputs
    mnkl = (args.M, args.N, args.K, args.L)
    atoms = (args.atom_m, args.atom_n, args.atom_k)
    cta = (args.cta_m, args.cta_n, args.cta_k)

    try:
        # Suppress verbose prints from the underlying module during run
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            elapsed = gemm_mod.run(
                a_major=args.a_major,
                b_major=args.b_major,
                c_major=args.c_major,
                ab_dtype=gemm_mod.cutlass.Float16,
                c_dtype=gemm_mod.cutlass.Float16,
                acc_dtype=gemm_mod.cutlass.Float32,
                mnkl=mnkl,
                atom_layout_mnk=atoms,
                warmup_iterations=args.warmup,
                iterations=args.iters,
                skip_ref_check=args.skip_ref_check,
                use_cold_l2=args.use_cold_l2,
                cta_tiler=cta,
                num_stages=args.stages,
            )
        print(json.dumps({"ok": True, "elapsed_us": float(elapsed)}))
    except AssertionError as e:
        # Treat assertion errors as "skip" (e.g., failed correctness or shape asserts)
        print(json.dumps({"ok": False, "kind": "skip", "error": str(e)}))
    except Exception as e:
        # Any other failure — compile/runtime
        print(json.dumps({"ok": False, "kind": "fail", "error": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
    main()
