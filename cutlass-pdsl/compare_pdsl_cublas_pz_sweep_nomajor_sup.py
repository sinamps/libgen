#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark CUTLASS CuTe-DSL Ampere TensorOp GEMM vs cuBLAS (PyTorch)
- Sweeps (M,N,K,L) problem sizes
- Records average time (us) and GFLOPs to CSV

Usage examples:
  python benchmark_cutlass_vs_cublas.py \
      --sizes "8192,8192,8192,1;4096,4096,4096,1" \
      --cutlass-module examples.ampere.tensorop_gemm \
      --out results.csv

  python benchmark_cutlass_vs_cublas.py \
      --sizes "8192,8192,8192,1" \
      --cutlass-file /path/to/examples/ampere/tensorop_gemm.py \
      --atom-layout 2,2,1 --iters 100 --warmup 2 --out results.csv
"""
import argparse
import csv
import importlib
import importlib.util
import os
import sys
from typing import List, Tuple

import torch


# --------------------------- helpers ---------------------------

def parse_sizes(s: str) -> List[Tuple[int, int, int, int]]:
    """
    Parse --sizes like "M,N,K,L;M,N,K,L;..."
    L is optional (defaults to 1) if given only three numbers.
    """
    sizes = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(x.strip()) for x in chunk.split(",")]
        if len(parts) == 3:
            M, N, K = parts
            L = 1
        elif len(parts) == 4:
            M, N, K, L = parts
        else:
            raise ValueError(f"Bad sizes chunk: {chunk} (expected M,N,K[,L])")
        sizes.append((M, N, K, L))
    return sizes


def load_cutlass_module(module_name: str = None, file_path: str = None):
    """
    Load the CUTLASS CuTe-DSL example so we can call its `run(...)`.
    Prefer module import if provided; otherwise load from file path.
    """
    if module_name:
        return importlib.import_module(module_name)
    if not file_path:
        raise ValueError("Provide --cutlass-module or --cutlass-file")
    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location("tensorop_gemm_cutlass", file_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def gflops(M: int, N: int, K: int, L: int, time_us: float) -> float:
    # GEMM FLOPs = 2*M*N*K per batch; multiply by L
    flops = 2.0 * M * N * K * L
    sec = time_us * 1e-6
    return (flops / sec) / 1e9 if sec > 0 else float("inf")


def torch_bmm_cublas_time_us(M: int, N: int, K: int, L: int,
                             dtype: torch.dtype,
                             warmup: int,
                             iters: int,
                             use_cold_l2: bool = False) -> float:
    """
    Time cuBLAS via PyTorch batched matmul.
    Shapes: (L, M, K) @ (L, K, N) -> (L, M, N)
    Uses CUDA events for accurate timing.
    If use_cold_l2=True, cycles through multiple workspaces to keep L2 cold.
    """
    device = torch.device("cuda")
    # Use half inputs; accumulation is handled by cuBLAS internally.
    A0 = torch.empty((L, M, K), dtype=dtype, device=device).uniform_(-1, 1)
    B0 = torch.empty((L, K, N), dtype=dtype, device=device).uniform_(-1, 1)

    # Prepare workspaces if cold L2 requested
    if use_cold_l2:
        # Heuristic: at least enough copies to exceed typical L2 sizes
        # (safe default: 16 copies; you can raise this if you like)
        workspace_count = 16
        As = [torch.empty_like(A0).copy_(A0) for _ in range(workspace_count)]
        Bs = [torch.empty_like(B0).copy_(B0) for _ in range(workspace_count)]
    else:
        workspace_count = 1
        As = [A0]
        Bs = [B0]

    # Warmup (and ensure kernels are compiled)
    for i in range(warmup):
        _ = torch.matmul(As[i % workspace_count], Bs[i % workspace_count])
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for i in range(iters):
        Ai = As[i % workspace_count]
        Bi = Bs[i % workspace_count]
        _ = torch.matmul(Ai, Bi)
    stop_evt.record()
    torch.cuda.synchronize()

    # elapsed_time returns milliseconds; convert to microseconds and average
    total_ms = start_evt.elapsed_time(stop_evt)
    avg_us = (total_ms * 1000.0) / max(1, iters)
    return avg_us

# --------------------------- main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark CUTLASS CuTe-DSL vs cuBLAS")
    parser.add_argument("--sizes", type=str, required=True,
                        help='Semicolon-separated list like "M,N,K,L;M,N,K,L;..." '
                             '(L optional; defaults to 1).')
    # How to find the CUTLASS example
    parser.add_argument("--cutlass-module", type=str, default=None,
                        help="Python module path to the example (e.g. examples.ampere.tensorop_gemm)")
    parser.add_argument("--cutlass-file", type=str, default=None,
                        help="Filesystem path to the example (e.g. /path/to/examples/ampere/tensorop_gemm.py)")

    # CUTLASS kernel options (match the example's CLI)
    parser.add_argument("--atom-layout", type=str, default="2,2,1",
                        help="Atom layout mnk, e.g. 2,2,1")
    parser.add_argument("--a-major", choices=["k", "m"], default="m")
    parser.add_argument("--b-major", choices=["k", "n"], default="n")
    parser.add_argument("--c-major", choices=["n", "m"], default="n")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--skip-ref-check", action="store_true")
    parser.add_argument("--cold-l2", action="store_true",
                        help="Use additional workspaces to keep L2 cold (both CUTLASS & cuBLAS)")

    # dtpyes: example supports ab=fp16, c=fp16, acc=fp32
    parser.add_argument("--dtype", choices=["fp16"], default="fp16",
                        help="Input/output dtype (example supports fp16)")
    parser.add_argument("--acc-dtype", choices=["fp32"], default="fp32",
                        help="Accumulator dtype for CUTLASS example (fp32)")

    parser.add_argument("--out", type=str, required=True, help="Output CSV path")

    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA GPU is required"

    # Make matmul behavior explicit (avoid TF32, etc.)
    torch.backends.cuda.matmul.allow_tf32 = False
    # For fp16 GEMM, cuBLAS will use tensor cores where possible.
    torch.set_float32_matmul_precision("high")

    sizes = parse_sizes(args.sizes)
    atom_layout_mnk = tuple(int(x.strip()) for x in args.atom_layout.split(","))
    if len(atom_layout_mnk) != 3:
        raise ValueError("--atom-layout must be like 2,2,1")

    # Load CUTLASS module
    cutlass_mod = load_cutlass_module(args.cutlass_module, args.cutlass_file)

    # Resolve CUTLASS dtypes from the module's type system
    cutlass_ab_dtype = cutlass_mod.cutlass.Float16
    cutlass_c_dtype = cutlass_mod.cutlass.Float16
    cutlass_acc_dtype = cutlass_mod.cutlass.Float32

    # Map to torch dtype for cuBLAS baseline
    torch_dtype = torch.float16

    # CSV setup
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "engine", "M", "N", "K", "L",
        "a_major", "b_major", "c_major",
        "ab_dtype", "c_dtype", "acc_dtype",
        "atom_layout_mnk",
        "warmup", "iters", "use_cold_l2",
        "avg_time_us", "gflops",
        "device_name", "sm_major", "sm_minor",
    ]
    device_name = torch.cuda.get_device_name()
    cc = torch.cuda.get_device_capability()
    sm_major, sm_minor = cc[0], cc[1]

    # Open CSV and append rows
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (M, N, K, L) in sizes:
            print(f"\n=== Problem: M={M} N={N} K={K} L={L} ===")

            # ---------------- CUTLASS (CuTe-DSL example) ----------------
            cutlass_time_us = cutlass_mod.run(
                a_major=args.a_major,
                b_major=args.b_major,
                c_major=args.c_major,
                ab_dtype=cutlass_ab_dtype,
                c_dtype=cutlass_c_dtype,
                acc_dtype=cutlass_acc_dtype,
                mnkl=(M, N, K, L),
                atom_layout_mnk=atom_layout_mnk,
                warmup_iterations=args.warmup,
                iterations=args.iters,
                skip_ref_check=args.skip_ref_check,
                use_cold_l2=args.cold_l2,
            )
            cutlass_gflops = gflops(M, N, K, L, cutlass_time_us)
            print(f"[CUTLASS] avg_time = {cutlass_time_us:.2f} us  |  {cutlass_gflops:.2f} GFLOPs")

            writer.writerow(dict(
                engine="cutlass_dsl",
                M=M, N=N, K=K, L=L,
                a_major=args.a_major, b_major=args.b_major, c_major=args.c_major,
                ab_dtype="fp16", c_dtype="fp16", acc_dtype="fp32",
                atom_layout_mnk=f"{atom_layout_mnk[0]},{atom_layout_mnk[1]},{atom_layout_mnk[2]}",
                warmup=args.warmup, iters=args.iters, use_cold_l2=int(args.cold_l2),
                avg_time_us=f"{cutlass_time_us:.3f}", gflops=f"{cutlass_gflops:.6f}",
                device_name=device_name, sm_major=sm_major, sm_minor=sm_minor,
            ))

            # ---------------- cuBLAS (PyTorch) ----------------
            cublas_time_us = torch_bmm_cublas_time_us(
                M=M, N=N, K=K, L=L,
                dtype=torch_dtype,
                warmup=args.warmup,
                iters=args.iters,
                use_cold_l2=args.cold_l2,
            )
            cublas_gflops = gflops(M, N, K, L, cublas_time_us)
            print(f"[cuBLAS ] avg_time = {cublas_time_us:.2f} us  |  {cublas_gflops:.2f} GFLOPs")

            writer.writerow(dict(
                engine="cublas",
                M=M, N=N, K=K, L=L,
                a_major="L,M,K", b_major="L,K,N", c_major="L,M,N",  # for reference (PyTorch shapes)
                ab_dtype="fp16", c_dtype="fp16", acc_dtype="(cuBLAS internal)",
                atom_layout_mnk="N/A",
                warmup=args.warmup, iters=args.iters, use_cold_l2=int(args.cold_l2),
                avg_time_us=f"{cublas_time_us:.3f}", gflops=f"{cublas_gflops:.6f}",
                device_name=device_name, sm_major=sm_major, sm_minor=sm_minor,
            ))

    print(f"\nDone. Wrote: {out_path}")


if __name__ == "__main__":
    main()
