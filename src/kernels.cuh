#pragma once

#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// ===========================================================================
// FP16 tiled GEMM on tensor cores (C = A * B)
//
//   Block tile : BM x BN        (one thread block)
//   K chunk    : BK             (streamed)
//   Warp grid  : WARP_M x WARP_N warps, each owning a 32 x 64 output sub-tile
//   MMA shape  : 16 x 16 x 16
//
// M, N, K must be multiples of BM / BN / BK (enforced by the host harness).
// ===========================================================================
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int WARP_M = 4;               // warp rows
constexpr int WARP_N = 2;               // warp cols
constexpr int NWARPS = WARP_M * WARP_N; // 8
constexpr int THREADS = NWARPS * 32;    // 256

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int MITER = (BM / WARP_M) / WMMA_M; // 32/16 = 2
constexpr int NITER = (BN / WARP_N) / WMMA_N; // 64/16 = 4
constexpr int KITER = BK / WMMA_K;            // 2

constexpr int A_TILE = BM * BK;  // halves in an A tile (128*32 = 4096)
constexpr int B_TILE = BK * BN;  // halves in a B tile (32*128 = 4096)
constexpr int A_F4 = A_TILE / 8; // float4 (8 halves) loads for an A tile
constexpr int B_F4 = B_TILE / 8;

// A tile: BM x BK, row-major in shared (ldm = BK). Source A is row-major M x K.
__device__ __forceinline__ void load_A_sync(half *As, const half *A, int K,
                                            int blockRow, int k0) {
  for (int fi = threadIdx.x; fi < A_F4; fi += THREADS) {
    const int row = fi / (BK / 8);
    const int col = (fi % (BK / 8)) * 8;
    *reinterpret_cast<float4 *>(&As[row * BK + col]) =
        *reinterpret_cast<const float4 *>(
            &A[(blockRow * BM + row) * K + k0 + col]);
  }
}

// B tile: BK x BN, row-major in shared (ldm = BN). Source B is row-major K x N.
__device__ __forceinline__ void load_B_sync(half *Bs, const half *B, int N,
                                            int blockCol, int k0) {
  for (int fi = threadIdx.x; fi < B_F4; fi += THREADS) {
    const int row = fi / (BN / 8);
    const int col = (fi % (BN / 8)) * 8;
    *reinterpret_cast<float4 *>(&Bs[row * BN + col]) =
        *reinterpret_cast<const float4 *>(
            &B[(k0 + row) * N + blockCol * BN + col]);
  }
}

__device__ __forceinline__ void load_A_cpasync(half *As, const half *A, int K,
                                               int blockRow, int k0) {
  for (int fi = threadIdx.x; fi < A_F4; fi += THREADS) {
    const int row = fi / (BK / 8);
    const int col = (fi % (BK / 8)) * 8;
    __pipeline_memcpy_async(&As[row * BK + col],
                            &A[(blockRow * BM + row) * K + k0 + col],
                            sizeof(float4));
  }
}

__device__ __forceinline__ void load_B_cpasync(half *Bs, const half *B, int N,
                                               int blockCol, int k0) {
  for (int fi = threadIdx.x; fi < B_F4; fi += THREADS) {
    const int row = fi / (BN / 8);
    const int col = (fi % (BN / 8)) * 8;
    __pipeline_memcpy_async(&Bs[row * BN + col],
                            &B[(k0 + row) * N + blockCol * BN + col],
                            sizeof(float4));
  }
}

using AccFrag =
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

__device__ __forceinline__ void mma_tile(const half *As, const half *Bs,
                                         int warpRow, int warpCol,
                                         AccFrag acc[MITER][NITER]) {
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag[MITER];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      b_frag[NITER];

#pragma unroll
  for (int kk = 0; kk < KITER; ++kk) {
#pragma unroll
    for (int mi = 0; mi < MITER; ++mi)
      wmma::load_matrix_sync(
          a_frag[mi],
          &As[(warpRow * (BM / WARP_M) + mi * WMMA_M) * BK + kk * WMMA_K], BK);
#pragma unroll
    for (int ni = 0; ni < NITER; ++ni)
      wmma::load_matrix_sync(
          b_frag[ni],
          &Bs[(kk * WMMA_K) * BN + warpCol * (BN / WARP_N) + ni * WMMA_N], BN);
#pragma unroll
    for (int mi = 0; mi < MITER; ++mi)
#pragma unroll
      for (int ni = 0; ni < NITER; ++ni)
        wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
  }
}

__device__ __forceinline__ void store_C(float *C, int N, int blockRow,
                                        int blockCol, int warpRow, int warpCol,
                                        AccFrag acc[MITER][NITER]) {
#pragma unroll
  for (int mi = 0; mi < MITER; ++mi)
#pragma unroll
    for (int ni = 0; ni < NITER; ++ni) {
      const int r = blockRow * BM + warpRow * (BM / WARP_M) + mi * WMMA_M;
      const int c = blockCol * BN + warpCol * (BN / WARP_N) + ni * WMMA_N;
      wmma::store_matrix_sync(&C[r * N + c], acc[mi][ni], N,
                              wmma::mem_row_major);
    }
}

__device__ __forceinline__ void init_acc(AccFrag acc[MITER][NITER]) {
#pragma unroll
  for (int mi = 0; mi < MITER; ++mi)
#pragma unroll
    for (int ni = 0; ni < NITER; ++ni)
      wmma::fill_fragment(acc[mi][ni], 0.0f);
}

// ===========================================================================
// (A) Baseline: synchronous shared-memory staging
// ===========================================================================
__global__ void gemm_sync(const half *__restrict__ A,
                          const half *__restrict__ B, float *__restrict__ C,
                          int M, int N, int K) {
  __shared__ half As[A_TILE];
  __shared__ half Bs[B_TILE];

  const int warp = threadIdx.x / 32;
  const int warpRow = warp / WARP_N, warpCol = warp % WARP_N;

  AccFrag acc[MITER][NITER];
  init_acc(acc);

  for (int k0 = 0; k0 < K; k0 += BK) {
    load_A_sync(As, A, K, blockIdx.y, k0);
    load_B_sync(Bs, B, N, blockIdx.x, k0);
    __syncthreads();

    mma_tile(As, Bs, warpRow, warpCol, acc);
    __syncthreads();
  }
  store_C(C, N, blockIdx.y, blockIdx.x, warpRow, warpCol, acc);
}

// ===========================================================================
// (B) Raw cp.async: multi-stage prefetch
// ===========================================================================
template <int STAGES>
__global__ void gemm_cpasync(const half *__restrict__ A,
                             const half *__restrict__ B, float *__restrict__ C,
                             int M, int N, int K) {
  __shared__ half As[STAGES][A_TILE];
  __shared__ half Bs[STAGES][B_TILE];

  const int warp = threadIdx.x / 32;
  const int warpRow = warp / WARP_N, warpCol = warp % WARP_N;
  const int nK = K / BK;

  auto fetch = [&](int stage, int kt) {
    load_A_cpasync(As[stage], A, K, blockIdx.y, kt * BK);
    load_B_cpasync(Bs[stage], B, N, blockIdx.x, kt * BK);
    __pipeline_commit();
  };

  AccFrag acc[MITER][NITER];
  init_acc(acc);

  int fetched = 0;
  const int prime = min(STAGES - 1, nK);
  for (; fetched < prime; ++fetched)
    fetch(fetched % STAGES, fetched);

  for (int kt = 0; kt < nK; ++kt) {
    if (fetched < nK) {
      fetch(fetched % STAGES, fetched);
      ++fetched;
    }
    // `fetched` groups have been committed; we are about to consume tile kt.
    // Leave (fetched - kt - 1) groups in flight so tile kt is complete.
    __pipeline_wait_prior(fetched - kt - 1);
    __syncthreads();

    mma_tile(As[kt % STAGES], Bs[kt % STAGES], warpRow, warpCol, acc);
    __syncthreads();
  }
  store_C(C, N, blockIdx.y, blockIdx.x, warpRow, warpCol, acc);
}

// ===========================================================================
// (C) cuda::pipeline: multi-stage abstraction (thread-local pipe -> cp.async)
// ===========================================================================
template <int STAGES>
__global__ void gemm_pipeline(const half *__restrict__ A,
                              const half *__restrict__ B, float *__restrict__ C,
                              int M, int N, int K) {
  __shared__ half As[STAGES][A_TILE];
  __shared__ half Bs[STAGES][B_TILE];

  auto pipe = cuda::make_pipeline();

  const int warp = threadIdx.x / 32;
  const int warpRow = warp / WARP_N, warpCol = warp % WARP_N;
  const int nK = K / BK;

  auto fetch = [&](int stage, int kt) {
    load_A_cpasync(As[stage], A, K, blockIdx.y, kt * BK);
    load_B_cpasync(Bs[stage], B, N, blockIdx.x, kt * BK);
    pipe.producer_commit();
  };

  AccFrag acc[MITER][NITER];
  init_acc(acc);

  int fetched = 0;
  const int prime = min(STAGES - 1, nK);
  for (; fetched < prime; ++fetched)
    fetch(fetched % STAGES, fetched);

  for (int kt = 0; kt < nK; ++kt) {
    if (fetched < nK) {
      fetch(fetched % STAGES, fetched);
      ++fetched;
    }
    pipe.consumer_wait();
    __syncthreads();

    mma_tile(As[kt % STAGES], Bs[kt % STAGES], warpRow, warpCol, acc);
    __syncthreads();
  }
  store_C(C, N, blockIdx.y, blockIdx.x, warpRow, warpCol, acc);
}
