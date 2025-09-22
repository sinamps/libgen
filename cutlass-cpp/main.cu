// Export CUTLASS Repo: export CUTLASS=/path/to/cutlass
// nvcc -O3 -std=c++17 -arch=sm_86 -I"$CUTLASS"/include -o cutlass_dyn_gemm main.cu

// main.cu
// CUTLASS Tensor Core GEMM with dynamic shapes (runtime M, N, K)
// A,B,C: FP16 (row-major), Accumulator: FP32
// RTX 3090 (Ampere, sm_86) example with correctness check vs CPU reference.
//
// Build (example):
//   nvcc -O3 -std=c++17 -arch=sm_86 -I/path/to/cutlass -o cutlass_dyn_gemm main.cu
//
// Run (examples):
//   ./cutlass_dyn_gemm                  # defaults M=1024 N=768 K=512
//   ./cutlass_dyn_gemm 123 777 1000    # arbitrary dynamic sizes

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <cstring>

// --- CUTLASS ---
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/arch/arch.h"

// Simple CUDA error check
#define CHECK_CUDA(call) do {                                 \
  cudaError_t _e = (call);                                     \
  if (_e != cudaSuccess) {                                     \
    std::cerr << "CUDA error " << cudaGetErrorString(_e)       \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";\
    std::exit(1);                                              \
  }                                                            \
} while(0)

using Elem = cutlass::half_t; // FP16
using Layout = cutlass::layout::RowMajor;

// Utility: convert float->half
inline cutlass::half_t f2h(float x) { return cutlass::half_t(x); }
// Utility: convert half->float
inline float h2f(cutlass::half_t h) { return static_cast<float>(h); }

// Fill host vector with random values in [-1,1]
template <typename T>
void fill_random(std::vector<T>& v, unsigned seed=1234) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = T(dist(rng));
}
// Specialization for half
template <>
void fill_random<cutlass::half_t>(std::vector<cutlass::half_t>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = f2h(dist(rng));
}

// CPU reference GEMM: D = alpha * A * B + beta * C
// A(MxK) row-major, B(KxN) row-major, C/D(MxN) row-major
void cpu_gemm_ref(int M, int N, int K,
                  float alpha,
                  const std::vector<Elem>& A, int lda,
                  const std::vector<Elem>& B, int ldb,
                  float beta,
                  const std::vector<Elem>& C, int ldc,
                  std::vector<Elem>& D, int ldd) {
  std::vector<float> acc(M * N, 0.0f);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a = h2f(A[m * lda + k]);
        float b = h2f(B[k * ldb + n]);
        sum += a * b;
      }
      float c = h2f(C[m * ldc + n]);
      float d = alpha * sum + beta * c;
      acc[m * N + n] = d;
    }
  }
  // Cast to half for D
  for (int i = 0; i < M * N; ++i) {
    D[i] = f2h(acc[i]);
  }
}

// Compare two FP16 matrices with tolerance (absolute + relative)
bool allclose_half(const std::vector<Elem>& x,
                   const std::vector<Elem>& y,
                   double atol = 5e-2, double rtol = 1e-2) {
  if (x.size() != y.size()) return false;
  size_t n = x.size();
  size_t mismatches = 0;
  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double a = h2f(x[i]);
    double b = h2f(y[i]);
    double diff = std::abs(a - b);
    double rel = diff / (std::abs(b) + 1e-8);
    if (diff > atol + rtol * std::abs(b)) {
      mismatches++;
      if (diff > max_abs) max_abs = diff;
      if (rel > max_rel) max_rel = rel;
    }
  }
  if (mismatches) {
    std::cerr << "[CHECK] FAILED: mismatches=" << mismatches
              << " / " << n
              << "  max_abs=" << max_abs
              << "  max_rel=" << max_rel << "\n";
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  // Parse dynamic sizes
  int M = 1024, N = 768, K = 512;
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }
  std::cout << "Running CUTLASS Tensor Core GEMM with dynamic sizes:\n";
  std::cout << "  M=" << M << "  N=" << N << "  K=" << K << "\n";

  // Leading dimensions for row-major
  int lda = K;
  int ldb = N;
  int ldc = N;
  int ldd = N;

  float alpha = 1.0f;
  float beta  = 1.0f;

  // Host buffers
  std::vector<Elem> hA(M * lda), hB(K * ldb), hC(M * ldc), hD(M * ldd);
  std::vector<Elem> hD_ref(M * ldd);

  fill_random(hA, 123);
  fill_random(hB, 456);
  fill_random(hC, 789);

  // Device allocations
  Elem *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr;
  CHECK_CUDA(cudaMalloc((void**)&dA, sizeof(Elem) * hA.size()));
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(Elem) * hB.size()));
  CHECK_CUDA(cudaMalloc((void**)&dC, sizeof(Elem) * hC.size()));
  CHECK_CUDA(cudaMalloc((void**)&dD, sizeof(Elem) * hD.size()));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(Elem) * hA.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(Elem) * hB.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeof(Elem) * hC.size(), cudaMemcpyHostToDevice));

  // -----------------------------
  // Define a CUTLASS TensorOp GEMM kernel for Ampere+
  // -----------------------------
  using Gemm = cutlass::gemm::device::Gemm<
      // Data types of A, B, C/D
      Elem, Layout,   // A
      Elem, Layout,   // B
      Elem, Layout,   // C/D
      float,          // Accumulator type
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, // Ampere family (sm_80+). OK for sm_86 GPUs like RTX 3090
      // Threadblock, Warp, and MMA instruction shapes
      cutlass::gemm::GemmShape<128, 128, 64>, // Threadblock tile
      cutlass::gemm::GemmShape<64, 64, 64>,   // Warp tile
      cutlass::gemm::GemmShape<16, 8, 8>      // Tensor Core MMA (FP16 on Ampere)
  >;

  // Arguments structure with runtime problem size and pointers/strides
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  typename Gemm::Arguments args{
      problem_size,
      {dA, lda},
      {dB, ldb},
      {dC, ldc},
      {dD, ldd},
      {alpha, beta}
  };

  // Check if the launch is supported (alignment, etc.)
  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::can_implement() says not supported: "
              << static_cast<int>(status) << "\n";
    return 1;
  }

  // Initialize (allocates internal workspace if needed)
  status = gemm_op.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::initialize() failed: " << static_cast<int>(status) << "\n";
    return 1;
  }

  // Launch
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::operator() launch failed: " << static_cast<int>(status) << "\n";
    return 1;
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back
  CHECK_CUDA(cudaMemcpy(hD.data(), dD, sizeof(Elem) * hD.size(), cudaMemcpyDeviceToHost));

  // Reference on CPU (float accumulation), then cast to half
  cpu_gemm_ref(M, N, K, alpha, hA, lda, hB, ldb, beta, hC, ldc, hD_ref, ldd);

  // Compare
  bool ok = allclose_half(hD, hD_ref, /*atol=*/5e-2, /*rtol=*/1e-2);
  if (ok) {
    std::cout << "[CHECK] PASS ✅ (within tolerance)\n";
  } else {
    std::cout << "[CHECK] FAIL ❌\n";
  }

  // Cleanup
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);

  return ok ? 0 : 2;
}
