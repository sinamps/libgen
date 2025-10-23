#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark CUTLASS CuTe-DSL Ampere TensorOp GEMM vs cuBLAS (PyTorch)

- EXACT majors parity with CUTLASS example (same permute/stride logic; no .contiguous()).
- C is written directly in the requested physical layout (no post-hoc transpose/copy).
- Optional correctness checks: GPU FP32 reference (matmul or einsum) and/or pairwise diff.
- Sweeps (M,N,K,L) sizes and writes times + GFLOPs to CSV.

Examples:
  python benchmark_cutlass_vs_cublas.py \
      --sizes "8192,8192,8192,1;4096,4096,4096,1" \
      --cutlass-module examples.ampere.tensorop_gemm \
      --out results.csv

  python benchmark_cutlass_vs_cublas.py \
      --sizes "8192,8192,8192,1" \
      --cutlass-file /path/to/examples/ampere/tensorop_gemm.py \
      --atom-layout 2,2,1 --iters 100 --warmup 2 --out results.csv \
      --check both --rtol 1e-3 --atol 1e-3 --seed 123
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
    flops = 2.0 * M * N * K * L
    sec = time_us * 1e-6
    return (flops / sec) / 1e9 if sec > 0 else float("inf")


# --------------------------- cuBLAS baseline with CUTLASS-parity majors ---------------------------

def _torch_like_cutlass_tensor(
    L: int, mode0: int, mode1: int, is_mode0_major: bool,
    dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Replicates CUTLASS example's create_and_permute_tensor torch side:
      if is_mode0_major: base (L, mode1, mode0) -> permute (2,1,0) => (mode0, mode1, L)
      else:              base (L, mode0, mode1) -> permute (1,2,0) => (mode0, mode1, L)
    NOTE: Do NOT call .contiguous() here — we want the same strides CUTLASS sees.
    """
    if is_mode0_major:
        base = torch.empty((L, mode1, mode0), dtype=torch.int32, device=device).random_(-2, 2)
        t = base.to(dtype=dtype).permute(2, 1, 0)
    else:
        base = torch.empty((L, mode0, mode1), dtype=torch.int32, device=device).random_(-2, 2)
        t = base.to(dtype=dtype).permute(1, 2, 0)
    return t


def _alloc_A_like_cutlass(L, M, K, dtype, device, a_major: str) -> torch.Tensor:
    return _torch_like_cutlass_tensor(L, M, K, is_mode0_major=(a_major == "m"), dtype=dtype, device=device)  # (M,K,L)


def _alloc_B_like_cutlass(L, K, N, dtype, device, b_major: str) -> torch.Tensor:
    return _torch_like_cutlass_tensor(L, N, K, is_mode0_major=(b_major == "n"), dtype=dtype, device=device)  # (N,K,L)


def _alloc_C_like_cutlass(L, M, N, dtype, device, c_major: str) -> torch.Tensor:
    return _torch_like_cutlass_tensor(L, M, N, is_mode0_major=(c_major == "m"), dtype=dtype, device=device)  # (M,N,L)


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
    cuBLAS timing fully mirroring CUTLASS tensor creation:
      A_torch: (M,K,L), B_torch: (N,K,L), C_torch: (M,N,L) with CUTLASS-equivalent strides.
    Compute with matmul on (L,M,K) @ (L,K,N) and write **directly** into C_torch storage via an out= view.
    """
    dev = torch.device("cuda")

    def make_triplet():
        A_torch = _alloc_A_like_cutlass(L, M, K, dtype, dev, a_major)  # (M,K,L)
        B_torch = _alloc_B_like_cutlass(L, K, N, dtype, dev, b_major)  # (N,K,L)
        C_torch = _alloc_C_like_cutlass(L, M, N, dtype, dev, c_major)  # (M,N,L)
        # Views for matmul:
        A_rm = A_torch.permute(2, 0, 1)  # (L,M,K)
        B_rm = B_torch.permute(2, 1, 0)  # (L,K,N)
        # Choose an out= view that maps back into C_torch's storage:
        out_view = C_torch.permute(2, 0, 1) if c_major == "n" else C_torch.permute(2, 1, 0)  # (L,M,N) or (L,N,M)
        return A_rm, B_rm, out_view

    if use_cold_l2:
        w = 16
        As, Bs, Cs = zip(*[make_triplet() for _ in range(w)])
    else:
        w = 1
        As, Bs, Cs = zip(make_triplet())

    # Warmup
    for i in range(warmup):
        torch.matmul(As[i % w], Bs[i % w], out=Cs[i % w])
    torch.cuda.synchronize()

    # Time with CUDA events
    start_evt = torch.cuda.Event(enable_timing=True)
    stop_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for i in range(iters):
        torch.matmul(As[i % w], Bs[i % w], out=Cs[i % w])
    stop_evt.record()
    torch.cuda.synchronize()

    total_ms = start_evt.elapsed_time(stop_evt)
    return (total_ms * 1000.0) / max(1, iters)


# --------------------------- Correctness (GPU-only, robust) ---------------------------

def _cute_make_tensor_like_example(cutlass_mod, L, mode0, mode1, is_mode0_major, dtype_cutlass):
    """
    Reproduce the example's tensor creation for correctness (torch side) and the CUTE tensor.
    Returns (cute_tensor, torch_tensor) where torch_tensor is (mode0, mode1, L).
    """
    from_dlpack = cutlass_mod.from_dlpack
    cutlass_torch = cutlass_mod.cutlass_torch

    shape = (L, mode1, mode0) if is_mode0_major else (L, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    torch_tensor = (
        torch.empty(*shape, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=cutlass_torch.dtype(dtype_cutlass))
        .permute(permute_order)
        .cuda()
    )
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
    Compile once for the shape, run once, return A(torch), B(torch), C(torch) in (M,K,L), (N,K,L), (M,N,L).
    """
    cute = cutlass_mod.cute

    mA, a_torch = _cute_make_tensor_like_example(
        cutlass_mod, L, M, K, is_mode0_major=(a_major == "m"), dtype_cutlass=ab_dtype
    )
    mB, b_torch = _cute_make_tensor_like_example(
        cutlass_mod, L, N, K, is_mode0_major=(b_major == "n"), dtype_cutlass=ab_dtype
    )
    mC, c_torch = _cute_make_tensor_like_example(
        cutlass_mod, L, M, N, is_mode0_major=(c_major == "m"), dtype_cutlass=c_dtype
    )

    tensor_op_gemm = cutlass_mod.TensorOpGemm(ab_dtype, c_dtype, acc_dtype, atom_layout_mnk)
    compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC)
    compiled_gemm(mA, mB, mC)

    return a_torch, b_torch, c_torch


def _fp32_reference(A_MKL: torch.Tensor, B_NKL: torch.Tensor, mode: str = "matmul") -> torch.Tensor:
    """
    Build an FP32 GPU reference C (M,N,L) from A (M,K,L) and B (N,K,L).
    mode='matmul' (default): uses (L,M,K) @ (L,K,N) in FP32.
    mode='einsum' : uses einsum("mkl,nkl->mnl") in FP32.
    """
    if mode == "einsum":
        C = torch.einsum(
            "mkl,nkl->mnl",
            A_MKL.to(torch.float32),
            B_NKL.to(torch.float32),
        )
        return C
    # matmul variant
    A_rm32 = A_MKL.permute(2, 0, 1).to(torch.float32)  # (L,M,K)
    B_rm32 = B_NKL.permute(2, 1, 0).to(torch.float32)  # (L,K,N)
    C_rm32 = torch.matmul(A_rm32, B_rm32)               # (L,M,N)
    return C_rm32.permute(1, 2, 0)                      # (M,N,L)


def check_correctness_gpu(
    cutlass_mod,
    M, N, K, L,
    a_major, b_major, c_major,
    ab_dtype, c_dtype, acc_dtype,
    atom_layout_mnk,
    mode: str = "ref32",  # none|pair|ref32|both
    ref_mode: str = "matmul",  # matmul|einsum
    rtol: float = 1e-5,
    atol: float = 1e-3,
):
    assert mode in {"none", "pair", "ref32", "both"}
    assert ref_mode in {"matmul", "einsum"}

    # CUTLASS result (and the exact A/B values it used)
    a_torch, b_torch, c_cutlass = compute_cutlass_output_once(
        cutlass_mod, a_major, b_major, c_major,
        ab_dtype, c_dtype, acc_dtype,
        M, N, K, L, atom_layout_mnk,
    )
    results = {"pair_ok": None, "ref_ok_cutlass": None, "ref_ok_cublas": None}

    # cuBLAS compute using the very same inputs (pairwise)
    if mode in {"pair", "both", "ref32"}:
        # Use matmul with FP16 inputs (like timing path) then permute back to (M,N,L)
        A_rm = a_torch.permute(2, 0, 1)  # (L,M,K)
        B_rm = b_torch.permute(2, 1, 0)  # (L,K,N)
        C_rm = torch.matmul(A_rm, B_rm)  # (L,M,N)
        c_cublas = C_rm.permute(1, 2, 0) # (M,N,L)
    else:
        c_cublas = None

    # GPU FP32 reference (robust)
    if mode in {"ref32", "both"}:
        ref32 = _fp32_reference(a_torch, b_torch, mode=ref_mode)
        torch.testing.assert_close(
            c_cutlass.to(torch.float32), ref32, rtol=rtol, atol=atol
        )
        results["ref_ok_cutlass"] = True

        torch.testing.assert_close(
            c_cublas.to(torch.float32), ref32, rtol=rtol, atol=atol
        )
        results["ref_ok_cublas"] = True

    if mode in {"pair", "both"}:
        torch.testing.assert_close(
            c_cutlass.to(torch.float32), c_cublas.to(torch.float32),
            rtol=rtol, atol=atol
        )
        results["pair_ok"] = True

    # Summaries (optional)
    def max_err(a, b):
        d = (a.to(torch.float32) - b.to(torch.float32)).abs()
        return float(d.max().item())

    if mode in {"ref32", "both"}:
        results["cutlass_vs_ref_max_abs"] = max_err(c_cutlass, ref32)
        results["cublas_vs_ref_max_abs"] = max_err(c_cublas, ref32)
    if mode in {"pair", "both"}:
        results["cutlass_vs_cublas_max_abs"] = max_err(c_cutlass, c_cublas)

    # Cleanup
    del a_torch, b_torch, c_cutlass
    if c_cublas is not None:
        del c_cublas
    if mode in {"ref32", "both"}:
        del ref32
    torch.cuda.synchronize()
    return results


# --------------------------- main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark CUTLASS CuTe-DSL vs cuBLAS")
    parser.add_argument("--sizes", type=str, required=True,
                        help='Semicolon-separated list "M,N,K,L;M,N,K,L;..." (L defaults to 1).')
    parser.add_argument("--cutlass-module", type=str, default=None,
                        help="Python module path (e.g. examples.ampere.tensorop_gemm)")
    parser.add_argument("--cutlass-file", type=str, default=None,
                        help="Filesystem path (e.g. /path/to/examples/ampere/tensorop_gemm.py)")

    # CUTLASS options (match example)
    parser.add_argument("--atom-layout", type=str, default="2,2,1")
    parser.add_argument("--a-major", choices=["k", "m"], default="m")
    parser.add_argument("--b-major", choices=["k", "n"], default="n")
    parser.add_argument("--c-major", choices=["n", "m"], default="n")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--skip-ref-check", action="store_true",
                        help="(CUTLASS perf path) skip example's own correctness")
    parser.add_argument("--cold-l2", action="store_true",
                        help="Cycle workspaces to approximate L2-cold (both engines)")

    # dtypes: example supports ab=fp16, c=fp16, acc=fp32
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--acc-dtype", choices=["fp32"], default="fp32")

    # Correctness
    parser.add_argument("--check", choices=["none", "ref32", "pair", "both"],
                        default="ref32",
                        help="Correctness: FP32 reference, pairwise, or both.")
    parser.add_argument("--ref-mode", choices=["matmul", "einsum"], default="matmul",
                        help="How to form the FP32 reference (default: matmul).")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-3)

    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed for reproducibility (torch cuda+cpu).")

    parser.add_argument("--out", type=str, required=True, help="Output CSV path")

    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA GPU is required"

    # Optional seeding
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Make matmul behavior explicit (avoid TF32 for FP32 cases)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("high")

    sizes = parse_sizes(args.sizes)
    atom_layout_mnk = tuple(int(x.strip()) for x in args.atom_layout.split(","))
    if len(atom_layout_mnk) != 3:
        raise ValueError("--atom-layout must be like 2,2,1")

    # Load CUTLASS module
    cutlass_mod = load_cutlass_module(args.cutlass_module, args.cutlass_file)

    # CUTLASS dtypes
    cutlass_ab_dtype = cutlass_mod.cutlass.Float16
    cutlass_c_dtype = cutlass_mod.cutlass.Float16
    cutlass_acc_dtype = cutlass_mod.cutlass.Float32

    # PyTorch dtype for cuBLAS baseline
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

            # Optional correctness (GPU-only). Do it first.
            if args.check != "none":
                try:
                    check = check_correctness_gpu(
                        cutlass_mod,
                        M, N, K, L,
                        args.a_major, args.b_major, args.c_major,
                        cutlass_ab_dtype, cutlass_c_dtype, cutlass_acc_dtype,
                        atom_layout_mnk,
                        mode=args.check, ref_mode=args.ref_mode,
                        rtol=args.rtol, atol=args.atol,
                    )
                    print("[CHECK]", check)
                except AssertionError as e:
                    print("[CHECK] FAILED:", e)
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
