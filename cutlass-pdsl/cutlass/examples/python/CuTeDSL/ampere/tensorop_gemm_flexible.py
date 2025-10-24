# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
from typing import Tuple, Type, Optional

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

"""
A parameterized dense GEMM (C = A * B) example for NVIDIA Ampere GPUs using CuTe DSL.

Goals:
- Preserve the original example's performance heuristics for the default tile (128x128x32, 3 stages)
- Allow experimenting with different CTA tile shapes and num_stages via CLI without breaking performance
- Keep coalesced access, shared-memory swizzles, cp.async stages, and epilogue vectorization logic intact

Constraints (same as original):
* Input/output dtypes: fp16
* Accumulator dtype: fp32
* Default tile: 128x128x32
* Atom layout's MNK must divide tile shape along MMA instruction shape (16x8x16)
* Contiguous dimensions of A/B/C must be at least 16B aligned
"""


class TensorOpGemm:
    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
        cta_tiler: Optional[Tuple[int, int, int]] = None,
        num_stages: Optional[int] = None,
    ):
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        self.cta_tiler = cta_tiler or (128, 128, 32)
        self.num_stages = int(num_stages) if num_stages is not None else 3
        # Keep the user-provided atom layout for reference (not used directly for tiling anymore)
        self.user_atom_layout_mnk = atom_layout_mnk
        self.bM, self.bN, self.bK = self.cta_tiler

        # Derive an effective warp tiling from CTA tile. Use up to 2 warps per M/N like the original.
        eff_m = min(2, max(1, self.bM // 64))
        # Keep two warps along N even for 64-wide tiles to preserve mapping consistency.
        eff_n = 2 if self.bN == 64 else min(2, max(1, self.bN // 64))
        eff_k = 1
        self.atom_layout_mnk = (eff_m, eff_n, eff_k)

        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32

        self.mma_inst_shape = (16, 8, 16)  # Ampere tensor op
        mmaM, mmaN, mmaK = self.mma_inst_shape

        # Core divisibility and pipeline constraints (fail fast and clearly)
        assert self.bM % (atom_lay_M * mmaM) == 0, "bM must be divisible by MMA tile * atom layout"
        assert self.bN % (atom_lay_N * mmaN) == 0, "bN must be divisible by MMA tile * atom layout"
        assert atom_lay_K == 1, "This example does not support atom layout K > 1"
        assert self.bK % mmaK == 0, "bK must be divisible by MMA instruction K"
        assert self.num_stages >= 3, "num_stages must be >= 3 for proper cp.async pipelining"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # Determine runtime layout majors for A/B/C
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        # Shared memory layouts for A/B (with defensive swizzle math)
        ab_copy_bits = 128  # keep 128b vectorized cp.async when possible
        sA_layout = self._make_smem_layout_AB(
            mA.element_type,
            self.a_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type,
            self.b_major_mode,
            ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )

        # C shared memory layout with a safe fallback for small-major tiles
        # Build both swizzled (fast) and non-swizzled (safe) layouts and pick at JIT time.
        sC_layout_swizzled = self._make_smem_layout_C_swizzled(
            mC.element_type,
            self.c_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[1]),
        )
        sC_layout_safe = self._make_smem_layout_C_safe(
            mC.element_type,
            self.c_major_mode,
            (self.cta_tiler[0], self.cta_tiler[1]),
        )

        bM, bN, _ = self.cta_tiler
        use_safe_c = (
            (self.c_major_mode == utils.LayoutEnum.ROW_MAJOR and bN <= 64)
            or (self.c_major_mode == utils.LayoutEnum.COL_MAJOR and bM <= 64)
        )
        sC_layout = sC_layout_safe if use_safe_c else sC_layout_swizzled

        # Shared memory size (C reuses A/B buffers in epilogue)
        smem_size = max(
            cute.size_in_bytes(mC.element_type, sC_layout),
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout),
        )

        # Global <-> Shared copies
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )

        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        # Epilogue copy atom with adaptive vector width
        c_copy_bits = self._choose_c_copy_bits(mC)
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type,
            num_bits_per_copy=c_copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
        )

        # MMA tiling
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        # Use the proven permutation heuristic from the original kernel, but with our derived atom layout.
        # This keeps warp-level register tiling consistent while allowing different CTA tiles.
        # Adapt N permutation: the original uses an extra ×2 on N for better s2r coalescing,
        # but for narrow N tiles (e.g., bN <= 64) this can misalign coverage. Use ×1 in that case.
        n_multiplier = 2 if self.bN >= 64 else 1
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * n_multiplier,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        # Grid and rasterization
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        rasterization_remap_grid_dim = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            cute.size(grid_dim[2]),
        )

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
            raster_factor,
            epilogue_op,
        ).launch(
            grid=rasterization_remap_grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # Thread/block indices and raster remap
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        offset_tile_x, offset_tile_y = self.raster_tile(bidx, bidy, rasterization_factor)
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            # Local tiles
            gA = cute.local_tile(
                mA[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, None, 1),
            )
            gB = cute.local_tile(
                mB[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(None, 1, 1),
            )
            gC = cute.local_tile(
                mC[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, 1, None),
            )

            # Shift first-k residual to front to simplify mainloop
            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(gA, mode=[2])
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            # Identity tensors for predication
            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB.layout.shape)
            cA = cute.local_tile(
                mcA[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, None, 1),
            )
            cB = cute.local_tile(
                mcB[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(None, 1, 1),
            )
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            # Shared memory buffers
            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
            sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
            sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout)

            # Thread-level partitions
            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)

            # Identity partitioning for predication shapes
            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            # Predication tensors for M/N bounds (per copy-atom granularity)
            tApA = cute.make_fragment(
                cute.make_layout(
                    (
                        tAgA.shape[0][1],
                        cute.size(tAgA, mode=[1]),
                        cute.size(tAgA, mode=[2]),
                    ),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            tBpB = cute.make_fragment(
                cute.make_layout(
                    (
                        tBsB.shape[0][1],
                        cute.size(tBsB, mode=[1]),
                        cute.size(tBsB, mode=[2]),
                    ),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range(tApA.shape[0], unroll_full=True):
                for m in cutlass.range(tApA.shape[1], unroll_full=True):
                    tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
            for rest_v in cutlass.range(tBpB.shape[0], unroll_full=True):
                for n in cutlass.range(tBpB.shape[1], unroll_full=True):
                    tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])

            # Prefetch prologue
            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()
            num_smem_stages = cute.size(tAsA, mode=[3])
            k_tile_count = cute.size(tAgA, mode=[3])
            k_tile_index = cutlass.Int32(0)

            for k in cutlass.range(tApA.shape[2], unroll_full=True):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, k, k_tile_index],
                        tAsA[None, None, k, 0],
                        pred=tApA[None, None, k],
                    )
            for k in cutlass.range(tBpB.shape[2], unroll_full=True):
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, k, k_tile_index],
                        tBsB[None, None, k, 0],
                        pred=tBpB[None, None, k],
                    )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, k_tile_index],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            # Tile MMA partitions and accumulators
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCsC = thr_mma.partition_C(sC)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            # s2r copy atoms and views
            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mA.element_type,
            )
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mB.element_type,
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1
            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_p[None, None, k_block_next],
                        tCrA_copy_view[None, None, k_block_next],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_p[None, None, k_block_next],
                        tCrB_copy_view[None, None, k_block_next],
                    )

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_A,
                                tAgA[None, None, None, k_tile_index],
                                tAsA[None, None, None, smem_pipe_write],
                                pred=tApA,
                            )

                    cute.gemm(
                        tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC
                    )

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, k_tile_index],
                                tBsB[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # Epilogue with fusion
            # Build counting tensor for predication along C before branching
            ceilM, ceilN, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
            mcC = cute.make_identity_tensor(
                (
                    cute.size(ceilM) * self.cta_tiler[0],
                    cute.size(ceilN) * self.cta_tiler[1],
                    1,
                )
            )
            cC = cute.local_tile(
                mcC[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, 1, None),
            )
            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            tCrD[None] = epilogue_op(tCrC.load()).to(self.c_dtype)
            # Small-N direct epilogue: for row-major C and bN<=64, store directly from compute-mapped C.
            use_direct_epilogue = (
                self.c_major_mode == utils.LayoutEnum.ROW_MAJOR and self.bN <= 64
            )
            if use_direct_epilogue:
                thr_mma = tiled_mma.get_slice(tidx)
                tCgC_direct = thr_mma.partition_C(gC)
                tCcC = thr_copy_C.partition_S(cC)
                tCpC = cute.make_fragment(
                    cute.make_layout(
                        (
                            tCgC_direct.shape[0][1],
                            cute.size(tCgC_direct, mode=[1]),
                            cute.size(tCgC_direct, mode=[2]),
                        ),
                        stride=(cute.size(tCgC_direct, mode=[1]), 1, 0),
                    ),
                    cutlass.Boolean,
                )
                for rest_v in cutlass.range(tCpC.shape[0], unroll_full=True):
                    for m in cutlass.range(tCpC.shape[1], unroll_full=True):
                        tCpC[rest_v, m, 0] = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mC.shape[0])
                tCrD_direct = cute.make_fragment_like(tCgC_direct, self.c_dtype)
                cute.autovec_copy(tCrD, tCrD_direct)
                for rest_v in cutlass.range(tCpC.shape[0], unroll_full=True):
                    for n in cutlass.range(tCpC.shape[2], unroll_full=True):
                        if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                            cute.copy(
                                tiled_copy_C,
                                tCrD_direct[None, None, n],
                                tCgC_direct[None, None, n],
                                pred=tCpC[None, None, n],
                            )
            else:
                # Default path: register -> shared, then vectorized GMEM store
                cute.autovec_copy(tCrD, tCsC)

            # Counting tensor for predication along C
            ceilM, ceilN, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
            mcC = cute.make_identity_tensor(
                (
                    cute.size(ceilM) * self.cta_tiler[0],
                    cute.size(ceilN) * self.cta_tiler[1],
                    1,
                )
            )
            cC = cute.local_tile(
                mcC[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, 1, None),
            )
            tCcC = thr_copy_C.partition_S(cC)

            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.arch.sync_threads()
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

            tCpC = cute.make_fragment(
                cute.make_layout(
                    (
                        tCgC_epilogue.shape[0][1],
                        cute.size(tCgC_epilogue, mode=[1]),
                        cute.size(tCgC_epilogue, mode=[2]),
                    ),
                    stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range(tCpC.shape[0], unroll_full=True):
                for m in cutlass.range(tCpC.shape[1], unroll_full=True):
                    tCpC[rest_v, m, 0] = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mC.shape[0])

            for rest_v in cutlass.range(tCpC.shape[0], unroll_full=True):
                for n in cutlass.range(tCpC.shape[2], unroll_full=True):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                        cute.copy(
                            tiled_copy_C,
                            tCrC_epilogue[None, None, n],
                            tCgC_epilogue[None, None, n],
                            pred=tCpC[None, None, n],
                        )
        return

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size
        den = max(1, copy_bits)
        q = max(1, (major_mode_size * dtype.width) // den)
        swizzle_bits = min(int(math.log2(q)), 3)
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_smem_layout_C_swizzled(self, dtype, major_mode, copy_bits, smem_tiler):
        bM, bN = smem_tiler
        major_mode_size = bN if major_mode == utils.LayoutEnum.ROW_MAJOR else bM
        den = max(1, copy_bits)
        q = max(1, (major_mode_size * dtype.width) // den)
        swizzle_bits = min(int(math.log2(q)), 3)
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4), 0, layout_atom_outer
        )
        # Keep original behavior: no swizzle for COL_MAJOR C
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer
            )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))

    def _make_smem_layout_C_safe(self, dtype, major_mode, smem_tiler):
        bM, bN = smem_tiler
        major_mode_size = bN if major_mode == utils.LayoutEnum.ROW_MAJOR else bM
        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        # Identity swizzle to avoid right_inverse issues while preserving the blocked layout
        layout_atom = cute.make_composed_layout(cute.make_swizzle(0, 3, 4), 0, layout_atom_outer)
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))

    def _make_gmem_tiled_copy_AB(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bK) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bN) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def raster_tile(self, i, j, f):
        new_i = i // f
        new_j = (i % f) + (j * f)
        return (new_i, new_j)

    def _choose_c_copy_bits(self, mC) -> int:
        # Prefer 128-bit vectorization when cleanly divisible; else 64-bit fallback.
        elem_bits = int(mC.element_type.width)
        bM, bN, _ = self.cta_tiler
        major_is_row = self.c_major_mode == utils.LayoutEnum.ROW_MAJOR
        major_extent = bN if major_is_row else bM
        for bits in (128, 64):
            copy_elems = bits // elem_bits
            if copy_elems <= 0:
                continue
            if major_extent % copy_elems != 0:
                continue
            threads_per_vec_strip = max(1, major_extent // copy_elems)
            if self.num_threads % threads_per_vec_strip != 0:
                continue
            q = (major_extent * elem_bits) // max(1, bits)
            if q < 1:
                continue
            return bits
        return 64


def run(
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    mnkl: Tuple[int, int, int, int],
    atom_layout_mnk: Tuple[int, int, int],
    warmup_iterations: int = 2,
    iterations: int = 100,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    cta_tiler: Tuple[int, int, int] = (128, 128, 32),
    num_stages: int = 3,
    **kwargs,
):
    print("Running Ampere tensor core GEMM example (flexible):")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {ab_dtype}, B dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Atoms layout: {atom_layout_mnk}")
    print(f"CTA tiler: {cta_tiler}")
    print(f"num_stages: {num_stages}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")
    M, N, K, L = mnkl

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else: (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        torch_tensor = (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(dtype))
            .permute(permute_order)
            .cuda()
        )
        # assume input is 16B aligned
        cute_tensor = (
            from_dlpack(torch_tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=(1 if not is_mode0_major else 0))
            .mark_compact_shape_dynamic(
                mode=(1 if not is_mode0_major else 0),
                stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
                divisibility=(128 // dtype.width),
            )
        )
        return cute_tensor, torch_tensor

    mA, a_torch = create_and_permute_tensor(L, M, K, a_major == "m", ab_dtype)
    mB, b_torch = create_and_permute_tensor(L, N, K, b_major == "n", ab_dtype)
    mC, c_torch = create_and_permute_tensor(L, M, N, c_major == "m", c_dtype)

    tensor_op_gemm = TensorOpGemm(
        ab_dtype,
        c_dtype,
        acc_dtype,
        atom_layout_mnk,
        cta_tiler=cta_tiler,
        num_stages=num_stages,
    )

    # Report the derived warp tiling used by the kernel
    print(f"Derived warp tiling (tC): {tensor_op_gemm.atom_layout_mnk} -> num_threads={tensor_op_gemm.num_threads}")

    print("Compiling kernel with cute.compile ...")
    compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC)

    print("Executing GEMM kernel...")

    if not skip_ref_check:
        ref = torch.einsum(
            "mkl,nkl->mnl",
            a_torch.to(dtype=torch.float32),
            b_torch.to(dtype=torch.float32),
        ).to(cutlass_torch.dtype(c_dtype))
        compiled_gemm(mA, mB, mC)
        print("Verifying results...")
        torch.testing.assert_close(c_torch.cpu(), ref.cpu(), atol=1e-03, rtol=1e-05)
        print("Results verified successfully!")

    def generate_tensors():
        a_workspace, _ = create_and_permute_tensor(L, M, K, a_major == "m", ab_dtype)
        b_workspace, _ = create_and_permute_tensor(L, N, K, b_major == "n", ab_dtype)
        c_workspace, _ = create_and_permute_tensor(L, M, N, c_major == "m", c_dtype)
        return testing.JitArguments(a_workspace, b_workspace, c_workspace)

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    avg_time_us = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=False,
    )

    return avg_time_us


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Flexible multistage block matmul with CuTe on GPU"
    )
    parser.add_argument("--mnkl", type=parse_comma_separated_ints, default=(112, 136, 40, 1))
    parser.add_argument("--atom_layout_mnk", type=parse_comma_separated_ints, default=(2, 2, 1))
    parser.add_argument("--cta_tiler", type=parse_comma_separated_ints, default=(128, 128, 32))
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument(
        "--ab_dtype", type=cutlass.dtype, choices=[cutlass.Float16], default=cutlass.Float16
    )
    parser.add_argument(
        "--acc_dtype", type=cutlass.dtype, choices=[cutlass.Float32], default=cutlass.Float32
    )
    parser.add_argument(
        "--c_dtype", type=cutlass.dtype, choices=[cutlass.Float16], default=cutlass.Float16
    )
    parser.add_argument("--a_major", choices=["k", "m"], default="m")
    parser.add_argument("--b_major", choices=["k", "n"], default="n")
    parser.add_argument("--c_major", choices=["n", "m"], default="n")
    parser.add_argument("--warmup_iterations", default=2, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()
    elapsed_time = run(
        args.a_major,
        args.b_major,
        args.c_major,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.mnkl,
        args.atom_layout_mnk,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        cta_tiler=tuple(args.cta_tiler),
        num_stages=int(args.num_stages),
    )
    print("PASS")
    print(f"Average execution time: {elapsed_time:.2f} us")
