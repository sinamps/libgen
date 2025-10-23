#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark CUTLASS CuTe-DSL Ampere TensorOp GEMM vs cuBLAS (PyTorch)
- Honors A/B/C majors (row/col) for both engines.
- Writes C in the requested physical layout (no post-hoc transpose).
- Sweeps (M,N,K,L) problem sizes and saves times + GFLOPs to CSV.
- NEW: Einsum-based correctness check for cuBLAS results (and integrates
       with CUTLASS module's own ref check).

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

# --------------------------- cuBLAS baseline with CUTLASS-parity majors ---------------------------

def _torch_like_cutlass_tensor(
    L: int, mode0: int, mode1: int, is_mode0_major: bool,
    dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Replicates CUTLASS example's create_and_permute_tensor:
      - if is_mode0_major: base shape (L, mode1, mode0) -> permute (2,1,0) => (mode0, mode1, L)
      - else:              base shape (L, mode0, mode1) -> permute (1,2,0) => (mode0, mode1, L)
    Returns a CUDA tensor with the same shape/strides semantics as CUTLASS's torch_tensor.
    """
    if is_mode0_major:
        base = torch.empty((L, mode1, mode0), dtype=torch.int32, device=device).random_(-2, 2)
        t = base.to(dtype=dtype).permute(2, 1, 0)  # (mode0, mode1, L)
    else:
        base = torch.empty((L, mode0, mode1), dtype=torch.int32, device=device).random_(-2, 2)
        t = base.to(dtype=dtype).permute(1, 2, 0)  # (mode0, mode1, L)
    return t


def _alloc_A_like_cutlass(L, M, K, dtype, device, a_major: str) -> torch.Tensor:
    # a_major == "m" means mode0=M is major (contiguous); else "k" means mode0=M is not major
    return _torch_like_cutlass_tensor(L, M, K, is_mode0_major=(a_major == "m"), dtype=dtype, device=device)


def _alloc_B_like_cutlass(L, K, N, dtype, device, b_major: str) -> torch.Tensor:
    # b_major == "n" means mode0=N is major; else "k" means not major
    return _torch_like_cutlass_tensor(L, N, K, is_mode0_major=(b_major == "n"), dtype=dtype, device=device)


def _alloc_C_like_cutlass(L, M, N, dtype, device, c_major: str) -> torch.Tensor:
    # c_major == "m" => mode0=M is major; c_major == "n" => not major
    return _torch_like_cutlass_tensor(L, M, N, is_mode0_major=(c_major == "m"), dtype=dtype, device=device)


def build_views_for_cublas(
    M, N, K, L, a_major, b_major, c_major, dtype, device
):
    """
    Allocates A/B/C with CUTLASS-parity majors, and returns:
      A_rm: (L,M,K), B_rm: (L,K,N), out_view: view writing directly into C storage
      plus C_physical: the (M,N,L) physical tensor matching the requested C major.
    """
    A_torch = _alloc_A_like_cutlass(L, M, K, dtype, device, a_major)  # (M,K,L)
    B_torch = _alloc_B_like_cutlass(L, K, N, dtype, device, b_major)  # (N,K,L)
    C_torch = _alloc_C_like_cutlass(L, M, N, dtype, device, c_major)  # (M,N,L)

    # cuBLAS-friendly views
    A_rm = A_torch.permute(2, 0, 1)  # (L,M,K)
    B_rm = B_torch.permute(2, 1, 0)  # (L,K,N)

    # 'out' view that writes directly into C storage
    out_view = C_torch.permute(2, 0, 1) if c_major == "n" else C_torch.permute(2, 1, 0)  # (L,M,N) or (L,N,M)
    return A_rm, B_rm, out_view, C_torch


def einsum_ref(LMK: torch.Tensor, LKN: torch.Tensor, want_view: str) -> torch.Tensor:
    """
    Compute FP32 reference with einsum and arrange to match the 'out' view.
    want_view: "lmn" or "lnm" (decided by c_major)
    """
    ref_lmn = torch.einsum("lmk,lkn->lmn", LMK.float(), LKN.float())
    if want_view == "lnm":
        return ref_lmn.permute(0, 2, 1).contiguous()
    return ref_lmn.contiguous()


def check_against_einsum(
    A_rm: torch.Tensor, B_rm: torch.Tensor, out_view: torch.Tensor, c_major: str,
    rtol: float = 1e-2, atol: float = 1e-2
):
    """
    Compare (cuBLAS) out_view with an einsum FP32 reference, allowing FP16 tolerances.
    """
    want = "lmn" if c_major == "n" else "lnm"
    ref = einsum_ref(A_rm, B_rm, want)  # FP32
    try:
        torch.testing.assert_close(out_view.float(), ref, rtol=rtol, atol=atol)
        print(f"[CHECK] PASS: cuBLAS vs einsum ({want}) within rtol={rtol}, atol={atol}")
    except AssertionError as e:
        # Helpful debug: max abs / rel diffs
        diff = (out_view.float() - ref).abs()
        max_abs = diff.max().item()
        rel = diff / (ref.abs() + 1e-12)
        max_rel = rel.max().item()
        print(f"[CHECK] FAILED: cuBLAS vs einsum ({want})")
        print(f"Max abs diff: {max_abs:.6g} | Max rel diff: {max_rel:.6g}")
        raise


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
      - A_torch: (M,K,L) with CUTLASS-equivalent strides
      - B_torch: (N,K,L) with CUTLASS-equivalent strides
      - C_torch: (M,N,L) with CUTLASS-equivalent strides (preallocated)
    We compute with matmul on (L,M,K) @ (L,K,N) and write results **directly into C_torch storage**
    via an 'out=' view that maps to (L, M, N) or (L, N, M) as appropriate.
    """
    dev = torch.device("cuda")

    # Build tensors exactly like the CUTLASS example does
    A_torch = _alloc_A_like_cutlass(L, M, K, dtype, dev, a_major)  # (M,K,L)
    B_torch = _alloc_B_like_cutlass(L, K, N, dtype, dev, b_major)  # (N,K,L)
    C_torch = _alloc_C_like_cutlass(L, M, N, dtype, dev, c_major)  # (M,N,L)

    # Views for cuBLAS-friendly shapes:
    #   A_rm: (L, M, K), B_rm: (L, K, N)
    A_rm = A_torch.permute(2, 0, 1)
    B_rm = B_torch.permute(2, 1, 0)

    # Choose an out= view that writes directly into C_torch's storage:
    out_view = C_torch.permute(2, 0, 1) if c_major == "n" else C_torch.permute(2, 1, 0)

    # Optional L2-cold workspaces (clones preserve strides)
    if use_cold_l2:
        w = 16
        As = [A_rm.clone() for _ in range(w)]
        Bs = [B_rm.clone() for _ in range(w)]
        Cs = [out_view.clone() for _ in range(w)]
    else:
        w = 1
        As = [A_rm]
        Bs = [B_rm]
        Cs = [out_view]

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
    print(f"\nsinamps cublas A shape and stride: {A_rm.shape, A_rm.stride()}")
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
    parser.add_argument("--skip-ref-check", action="store_true",
                        help="Pass-through to CUTLASS module; cuBLAS einsum check still runs unless --skip-einsum-check")
    parser.add_argument("--cold-l2", action="store_true",
                        help="Cycle workspaces to approximate L2-cold behavior (both engines)")

    # dtypes: example supports ab=fp16, c=fp16, acc=fp32
    parser.add_argument("--dtype", choices=["fp16"], default="fp16",
                        help="Input/output dtype (example supports fp16)")
    parser.add_argument("--acc-dtype", choices=["fp32"], default="fp32",
                        help="Accumulator dtype for CUTLASS example (fp32)")

    parser.add_argument("--out", type=str, required=True, help="Output CSV path")

    # NEW: controls for our einsum check
    parser.add_argument("--skip-einsum-check", action="store_true",
                        help="Skip the cuBLAS vs einsum correctness check")
    parser.add_argument("--rtol", type=float, default=1e-2, help="rtol for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="atol for correctness check")

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
    # cutlass_acc_dtype = cutlass_mod.cutlass.Float16

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
            # Note: If skip_ref_check=False, most examples already validate against a PyTorch ref.
            # We still do our own cuBLAS vs einsum check below.
            try:
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
                # print(f"sinamps c_major:{args.c_major}")
            except AssertionError as e:
                # Surface CUTLASS module's own check failure clearly
                print("[CUTLASS] Reference check FAILED inside module:")
                raise
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

            # ---------------- NEW: einsum correctness check (cuBLAS) ----------------
            if not args.skip_einsum_check:
                dev = torch.device("cuda")
                A_rm_chk, B_rm_chk, out_view_chk, _ = build_views_for_cublas(
                    M, N, K, L, args.a_major, args.b_major, args.c_major, torch_dtype, dev
                )
                # Single forward pass (not timed) on these check tensors
                torch.matmul(A_rm_chk, B_rm_chk, out=out_view_chk)
                check_against_einsum(
                    A_rm_chk, B_rm_chk, out_view_chk, args.c_major,
                    rtol=args.rtol, atol=args.atol
                )

            # If your CUTLASS module later exposes a "run_and_return_tensors" API,
            # you can also compare CUTLASS's actual output against einsum here:
            # if hasattr(cutlass_mod, "run_and_return_tensors"):
            #     A_cut, B_cut, C_cut, _time_us = cutlass_mod.run_and_return_tensors(...)
            #     # Make views matching (L,M,K) and (L,K,N) from those A/B, then compare
            #     # out_view(C_cut) to einsum_ref(...).

    print(f"\nDone. Wrote: {out_path}")


if __name__ == "__main__":
    main()
