#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark CUTLASS CuTe-DSL Ampere TensorOp GEMM vs cuBLAS (PyTorch)
- Honors A/B/C majors (row/col) for both engines.
- Writes C in the requested physical layout (no post-hoc transpose).
- Optional correctness checks against GPU FP32 reference and/or pairwise diff.
- Sweeps (M,N,K,L) problem sizes and saves times + GFLOPs to CSV.

Examples:
  python benchmark_cutlass_vs_cublas.py \
      --sizes "8192,8192,8192,1;4096,4096,4096,1" \
      --cutlass-module examples.ampere.tensorop_gemm \
      --out results.csv

  python benchmark_cutlass_vs_cublas.py \
      --sizes "8192,8192,8192,1" \
      --cutlass-file /path/to/examples/ampere/tensorop_gemm.py \
      --atom-layout 2,2,1 --iters 100 --warmup 2 --out results.csv \
      --check both --rtol 1e-3 --atol 1e-3
"""
import argparse
import csv
import importlib
import importlib.util
import os
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
    Load the CUTLASS CuTe-DSL example so we can call its components.
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
        A = base.transpose(-1, -2)  # (L,M,K) with col-major-like strides
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
        B = base.transpose(-1, -2)  # (L,K,N) with col-major-like strides
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
        C = base.transpose(-1, -2)  # (L,M,N) with col-major-like strides
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

    # Warmup
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
    return (total_ms * 1000.0) / max(1, iters)


# --------------------------- Correctness (fast, GPU-only) ---------------------------

def _cute_make_tensor_like_example(cutlass_mod, L, mode0, mode1, is_mode0_major, dtype_cutlass):
    """
    Reproduce the example's tensor creation for correctness:
      - creates a torch tensor of shape (mode0, mode1, L) with values
      - creates a CuTe tensor from dlpack with layout metadata
    Returns (cute_tensor, torch_tensor)
    """
    from_dlpack = cutlass_mod.from_dlpack
    cutlass_torch = cutlass_mod.cutlass_torch
    cute = cutlass_mod.cute

    # Match example's logic
    shape = (L, mode1, mode0) if is_mode0_major else (L, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    torch_tensor = (
        torch.empty(*shape, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=cutlass_torch.dtype(dtype_cutlass))
        .permute(permute_order)
        .cuda()
    )
    # Assume input is 16B aligned — mark layout
    cute_tensor = (
        from_dlpack(torch_tensor, assumed_align=16)
        .mark_layout_dynamic(leading_dim=(1 if not is_mode0_major else 0))
        .mark_compact_shape_dynamic(
            mode=(1 if not is_mode0_major else 0),
            stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
            divisibility=(128 // dtype_cutlass.width),
        )
    )
    return cute_tensor, torch_tensor


def compute_cutlass_output_once(
    cutlass_mod,
    a_major: str, b_major: str, c_major: str,
    ab_dtype, c_dtype, acc_dtype,
    M, N, K, L, atom_layout_mnk,
):
    """
    Compile once for the shape, run once, return C (torch) shaped (M,N,L).
    """
    cutlass = cutlass_mod.cutlass
    cute = cutlass_mod.cute

    # Create A,B,C like the example does
    mA, a_torch = _cute_make_tensor_like_example(
        cutlass_mod, L, M, K, is_mode0_major=(a_major == "m"), dtype_cutlass=ab_dtype
    )
    mB, b_torch = _cute_make_tensor_like_example(
        cutlass_mod, L, N, K, is_mode0_major=(b_major == "n"), dtype_cutlass=ab_dtype
    )
    mC, c_torch = _cute_make_tensor_like_example(
        cutlass_mod, L, M, N, is_mode0_major=(c_major == "m"), dtype_cutlass=c_dtype
    )

    # Build and run kernel like the example
    tensor_op_gemm = cutlass_mod.TensorOpGemm(ab_dtype, c_dtype, acc_dtype, atom_layout_mnk)
    compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC)
    compiled_gemm(mA, mB, mC)

    # c_torch now has the result in (M,N,L)
    return a_torch, b_torch, c_torch


def compute_cublas_output_once_from_AT_BN(
    a_torch_MKL, b_torch_NKL,  # shapes (M,K,L), (N,K,L) like example holds
    # We compute standard row-major matmul on (L,M,K) @ (L,K,N)
):
    """
    Produce cuBLAS result (M,N,L) from the same random inputs the example used.
    We don't emulate majors here — this is for numerical equivalence only.
    """
    # Reorder to (L,M,K) and (L,K,N)
    A_rm = a_torch_MKL.permute(2, 0, 1).contiguous()
    B_rm = b_torch_NKL.permute(2, 1, 0).contiguous()
    C_rm = torch.matmul(A_rm, B_rm)  # (L,M,N)
    C_MNL = C_rm.permute(1, 2, 0).contiguous()
    return C_MNL


def check_correctness_gpu(
    cutlass_mod,
    M, N, K, L,
    a_major, b_major, c_major,
    ab_dtype, c_dtype, acc_dtype,
    atom_layout_mnk,
    mode: str = "ref32",  # none|pair|ref32|both
    rtol: float = 1e-5,
    atol: float = 1e-3,
):
    """
    - Always compute CUTLASS once
    - If 'pair' or 'both': compute cuBLAS once (with same inputs) and compare CUTLASS vs cuBLAS
    - If 'ref32' or 'both': compute FP32 GPU reference from the same inputs and compare both to it
    Returns a dict with basic error stats and booleans.
    """
    assert mode in {"none", "pair", "ref32", "both"}
    device = torch.device("cuda")

    # CUTLASS result (and the exact A/B values it used)
    a_torch, b_torch, c_cutlass = compute_cutlass_output_once(
        cutlass_mod, a_major, b_major, c_major,
        ab_dtype, c_dtype, acc_dtype,
        M, N, K, L, atom_layout_mnk,
    )
    results = {"pair_ok": None, "ref_ok_cutlass": None, "ref_ok_cublas": None}

    # Optional: cuBLAS compute using the very same inputs
    if mode in {"pair", "both", "ref32"}:
        c_cublas = compute_cublas_output_once_from_AT_BN(a_torch, b_torch)
    else:
        c_cublas = None

    # Optional: high-precision reference on GPU (FP32)
    if mode in {"ref32", "both"}:
        ref32 = torch.einsum(
            "mkl,nkl->mnl",
            a_torch.to(dtype=torch.float32),
            b_torch.to(dtype=torch.float32),
        )
        # Compare CUTLASS → ref32
        torch.testing.assert_close(
            c_cutlass.to(torch.float32), ref32, rtol=rtol, atol=atol
        )
        results["ref_ok_cutlass"] = True
        # Compare cuBLAS → ref32
        torch.testing.assert_close(
            c_cublas.to(torch.float32), ref32, rtol=rtol, atol=atol
        )
        results["ref_ok_cublas"] = True

    # Optional: pairwise CUTLASS vs cuBLAS
    if mode in {"pair", "both"}:
        torch.testing.assert_close(
            c_cutlass.to(torch.float32), c_cublas.to(torch.float32),
            rtol=rtol, atol=atol
        )
        results["pair_ok"] = True

    # Return a tiny summary (max errors) for logging
    def max_err(a, b):
        d = (a.to(torch.float32) - b.to(torch.float32)).abs()
        return float(d.max().item())

    if mode in {"ref32", "both"}:
        results["cutlass_vs_ref_max_abs"] = max_err(c_cutlass, ref32)
        results["cublas_vs_ref_max_abs"] = max_err(c_cublas, ref32)
    if mode in {"pair", "both"}:
        results["cutlass_vs_cublas_max_abs"] = max_err(c_cutlass, c_cublas)

    # Clean up to keep memory fresh for perf runs
    del a_torch, b_torch, c_cutlass, c_cublas
    if mode in {"ref32", "both"}:
        del ref32
    torch.cuda.synchronize()
    return results


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
    parser.add_argument("--skip-ref-check", action="store_true",
                        help="(CUTLASS side only) skip example's own correctness in perf path")
    parser.add_argument("--cold-l2", action="store_true",
                        help="Cycle workspaces to approximate L2-cold behavior (both engines)")

    # dtypes: example supports ab=fp16, c=fp16, acc=fp32
    parser.add_argument("--dtype", choices=["fp16"], default="fp16",
                        help="Input/output dtype (example supports fp16)")
    parser.add_argument("--acc-dtype", choices=["fp32"], default="fp32",
                        help="Accumulator dtype for CUTLASS example (fp32)")

    parser.add_argument("--check", choices=["none", "ref32", "pair", "both"],
                        default="ref32",
                        help="Correctness checking mode: FP32 GPU reference, pairwise, or both.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tol for checks")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tol for checks")

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

            # Optional correctness (GPU-only, fast). Do it first to avoid
            # polluting perf runs with extra allocations (we sync anyway).
            if args.check != "none":
                try:
                    check = check_correctness_gpu(
                        cutlass_mod,
                        M, N, K, L,
                        args.a_major, args.b_major, args.c_major,
                        cutlass_ab_dtype, cutlass_c_dtype, cutlass_acc_dtype,
                        atom_layout_mnk,
                        mode=args.check,
                        rtol=args.rtol, atol=args.atol,
                    )
                    print("[CHECK]", check)
                except AssertionError as e:
                    print("[CHECK] FAILED:", e)
                    # You can choose to continue or raise; here we raise.
                    raise

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
