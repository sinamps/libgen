#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark CUTLASS CuTe-DSL Ampere TensorOp GEMM vs cuBLAS (PyTorch)
- Honors A/B/C majors (row/col) for both engines.
- Writes C in the requested physical layout (no post-hoc transpose).
- Sweeps (M,N,K,L) problem sizes and saves times + GFLOPs to CSV.

Examples:
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


# --------------------------- cuBLAS baseline with full majors parity ---------------------------

def _alloc_with_major_for_A(L, M, K, dtype, device, a_major: str):
    """
    Return a tensor shaped (L, M, K) whose *strides* emulate:
      - a_major == "m": row-major MxK (contiguous along K)
      - a_major == "k": col-major MxK (contiguous along M)
    """
    if a_major == "m":
        A = torch.empty((L, M, K), dtype=dtype, device=device).uniform_(-1, 1)
    else:  # "k" => col-major MxK
        base = torch.empty((L, K, M), dtype=dtype, device=device).uniform_(-1, 1)
        A = base.transpose(-1, -2)  # shape (L, M, K), strides like col-major
    return A


def _alloc_with_major_for_B(L, K, N, dtype, device, b_major: str):
    """
    Return a tensor shaped (L, K, N) whose *strides* emulate:
      - b_major == "n": row-major KxN (contiguous along N)
      - b_major == "k": col-major KxN (contiguous along K)
    """
    if b_major == "n":
        B = torch.empty((L, K, N), dtype=dtype, device=device).uniform_(-1, 1)
    else:  # "k" => col-major KxN
        base = torch.empty((L, N, K), dtype=dtype, device=device).uniform_(-1, 1)
        B = base.transpose(-1, -2)  # shape (L, K, N), col-major strides
    return B


def _alloc_out_with_major_for_C(L, M, N, dtype, device, c_major: str):
    """
    Return an output buffer (shape (L, M, N)) with strides that emulate:
      - c_major == "n": row-major MxN (contiguous along N)
      - c_major == "m": col-major MxN (contiguous along M)
    Implemented by creating a base contiguous tensor and taking a transpose view
    to get the desired (L, M, N) strides (no copies).
    """
    if c_major == "n":
        C = torch.empty((L, M, N), dtype=dtype, device=device)
    else:  # "m" => col-major MxN
        base = torch.empty((L, N, M), dtype=dtype, device=device)
        C = base.transpose(-1, -2)  # shape (L, M, N), col-major strides
    return C


def torch_cublas_time_us_parity(
    M, N, K, L,
    a_major: str,  # "m" or "k"
    b_major: str,  # "n" or "k"
    c_major: str,  # "n" or "m"
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    use_cold_l2: bool = False,
) -> float:
    """
    Majors-correct cuBLAS timing:
    - A: shape (L,M,K) with strides emulating requested major
    - B: shape (L,K,N) with strides emulating requested major
    - C: preallocated out= buffer with strides emulating requested major
    - Uses CUDA events for timing; optionally cycles workspaces for L2-cold.

    NOTE: We pass `out=` to torch.matmul so the kernel writes *directly* into
    the desired physical layout of C (no post-hoc transpose/copy).
    """
    dev = torch.device("cuda")

    A = _alloc_with_major_for_A(L, M, K, dtype, dev, a_major)
    B = _alloc_with_major_for_B(L, K, N, dtype, dev, b_major)
    C_out = _alloc_out_with_major_for_C(L, M, N, dtype, dev, c_major)

    # Prepare workspaces if requested (clone preserves strides)
    if use_cold_l2:
        w = 16
        As = [A.clone() for _ in range(w)]
        Bs = [B.clone() for _ in range(w)]
        Cs = [C_out.clone() for _ in range(w)]
    else:
        w = 1
        As = [A]
        Bs = [B]
        Cs = [C_out]

    # Warmup (ensure kernels/materialization are ready)
    for i in range(warmup):
        torch.matmul(As[i % w], Bs[i % w], out=Cs[i % w])
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for i in range(iters):
        torch.matmul(As[i % w], Bs[i % w], out=Cs[i % w])
    stop_evt.record()
    torch.cuda.synchronize()

    total_ms = start_evt.elapsed_time(stop_evt)
    print(f"\nsinamps cublas: {A.shape, A.stride()}")
    return (total_ms * 1000.0) / max(1, iters)


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
                        help="Cycle workspaces to approximate L2-cold behavior (both engines)")

    # dtypes: example supports ab=fp16, c=fp16, acc=fp32
    parser.add_argument("--dtype", choices=["fp16"], default="fp16",
                        help="Input/output dtype (example supports fp16)")
    parser.add_argument("--acc-dtype", choices=["fp32"], default="fp32",
                        help="Accumulator dtype for CUTLASS example (fp32)")

    parser.add_argument("--out", type=str, required=True, help="Output CSV path")

    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA GPU is required"

    # Make matmul behavior explicit (avoid TF32, etc.)
    torch.backends.cuda.matmul.allow_tf32 = False
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

            # ---------------- cuBLAS (PyTorch, majors-parity) ----------------
            cublas_time_us = torch_cublas_time_us_parity(
                M=M, N=N, K=K, L=L,
                a_major=args.a_major,
                b_major=args.b_major,
                c_major=args.c_major,
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
                a_major=args.a_major, b_major=args.b_major, c_major=args.c_major,
                ab_dtype="fp16", c_dtype="fp16", acc_dtype="(FP32 accumulate)",
                atom_layout_mnk="N/A",
                warmup=args.warmup, iters=args.iters, use_cold_l2=int(args.cold_l2),
                avg_time_us=f"{cublas_time_us:.3f}", gflops=f"{cublas_gflops:.6f}",
                device_name=device_name, sm_major=sm_major, sm_minor=sm_minor,
            ))

    print(f"\nDone. Wrote: {out_path}")


if __name__ == "__main__":
    main()
