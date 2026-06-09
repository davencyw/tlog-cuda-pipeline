#pragma once

#include "kernels.cuh"

#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>

// Hopper TMA pipelined GEMM: bulk 2D tile copies via cp.async.bulk.tensor +
// mbarrier completion, same WMMA compute path as the cp.async variants.

inline void tma_check(CUresult err, const char *what) {
  if (err == CUDA_SUCCESS)
    return;
  const char *msg = nullptr;
  cuGetErrorString(err, &msg);
  std::fprintf(stderr, "CUDA driver error in %s: %s\n", what,
               msg ? msg : "unknown");
  std::exit(EXIT_FAILURE);
}

// Row-major half matrix with inner dimension dim0 and outer dimension dim1.
inline CUtensorMap encode_tma_2d(void *global_addr, uint64_t dim0,
                                 uint64_t dim1, uint32_t box0, uint32_t box1) {
  CUtensorMap map{};
  const uint64_t global_dim[2] = {dim0, dim1};
  const uint64_t global_stride[1] = {dim0 * sizeof(half)};
  const uint32_t box_dim[2] = {box0, box1};
  const uint32_t elem_stride[2] = {1, 1};
  tma_check(cuTensorMapEncodeTiled(
                &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, global_addr,
                global_dim, global_stride, box_dim, elem_stride,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
            "cuTensorMapEncodeTiled");
  return map;
}

inline CUtensorMap encode_tma_a(const half *A, int M, int K) {
  return encode_tma_2d(const_cast<half *>(A), static_cast<uint64_t>(K),
                       static_cast<uint64_t>(M), BK, BM);
}

inline CUtensorMap encode_tma_b(const half *B, int K, int N) {
  return encode_tma_2d(const_cast<half *>(B), static_cast<uint64_t>(N),
                       static_cast<uint64_t>(K), BN, BK);
}

template <int STAGES>
__global__ void gemm_tma(__grid_constant__ const CUtensorMap tensor_a,
                         __grid_constant__ const CUtensorMap tensor_b,
                         float *__restrict__ C, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ half As[STAGES][A_TILE];
  __shared__ half Bs[STAGES][B_TILE];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bar_a[STAGES];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar_b[STAGES];
  __shared__ uint32_t phase_a[STAGES];
  __shared__ uint32_t phase_b[STAGES];

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      init(&bar_a[s], 1);
      init(&bar_b[s], 1);
      phase_a[s] = 0;
      phase_b[s] = 0;
    }
  }
  __syncthreads();

  const int warp = threadIdx.x / 32;
  const int warpRow = warp / WARP_N, warpCol = warp % WARP_N;
  const int nK = K / BK;
  constexpr uint32_t A_BYTES = A_TILE * sizeof(half);
  constexpr uint32_t B_BYTES = B_TILE * sizeof(half);

  auto tma_fetch = [&](int stage, int kt) {
    if (threadIdx.x == 0) {
      const int32_t coord_a[2] = {static_cast<int32_t>(kt * BK),
                                  static_cast<int32_t>(blockIdx.y * BM)};
      const int32_t coord_b[2] = {static_cast<int32_t>(blockIdx.x * BN),
                                  static_cast<int32_t>(kt * BK)};
      auto *bar_a_h = reinterpret_cast<uint64_t *>(
          cuda::device::barrier_native_handle(bar_a[stage]));
      auto *bar_b_h = reinterpret_cast<uint64_t *>(
          cuda::device::barrier_native_handle(bar_b[stage]));
      (void)cuda::device::barrier_arrive_tx(bar_a[stage], 1, A_BYTES);
      cuda::ptx::cp_async_bulk_tensor(
          cuda::ptx::space_cluster, cuda::ptx::space_global, As[stage],
          &tensor_a, coord_a, bar_a_h);
      (void)cuda::device::barrier_arrive_tx(bar_b[stage], 1, B_BYTES);
      cuda::ptx::cp_async_bulk_tensor(
          cuda::ptx::space_cluster, cuda::ptx::space_global, Bs[stage],
          &tensor_b, coord_b, bar_b_h);
    }
  };

  auto tma_wait = [&](int stage) {
    __syncthreads();
    auto *bar_a_h = reinterpret_cast<uint64_t *>(
        cuda::device::barrier_native_handle(bar_a[stage]));
    auto *bar_b_h = reinterpret_cast<uint64_t *>(
        cuda::device::barrier_native_handle(bar_b[stage]));
    while (!cuda::ptx::mbarrier_try_wait_parity(bar_a_h, phase_a[stage])) {
    }
    while (!cuda::ptx::mbarrier_try_wait_parity(bar_b_h, phase_b[stage])) {
    }
    if (threadIdx.x == 0) {
      phase_a[stage] ^= 1;
      phase_b[stage] ^= 1;
    }
    __syncthreads();
  };

  AccFrag acc[MITER][NITER];
  init_acc(acc);

  int fetched = 0;
  const int prime = min(STAGES - 1, nK);
  for (; fetched < prime; ++fetched)
    tma_fetch(fetched % STAGES, fetched);

  // Wait -> MMA -> prefetch. A pre-wait fetch to the stage we are about to
  // consume corrupts the mbarrier when (kt % STAGES) == (fetched % STAGES).
  for (int kt = 0; kt < nK; ++kt) {
    tma_wait(kt % STAGES);
    mma_tile(As[kt % STAGES], Bs[kt % STAGES], warpRow, warpCol, acc);
    __syncthreads();
    if (fetched < nK) {
      tma_fetch(fetched % STAGES, fetched);
      ++fetched;
    }
  }
  store_C(C, N, blockIdx.y, blockIdx.x, warpRow, warpCol, acc);
#endif
}