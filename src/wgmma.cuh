#pragma once

#include "common.cuh"
#include "kernels.cuh"

#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Hopper: 128B-swizzled TMA + WGMMA m64n128k16 f16 acc (TN: A MxK, B NxK K-major).
// Warp-specialized: warpgroup 0 produces (TMA), warpgroup 1 consumes (WGMMA).
// STAGES=1: single tile buffer, synchronous load-then-compute per K-slab.
// STAGES>1: multi-buffer software pipeline overlapping TMA with WGMMA.

constexpr int WGMMA_BK = 64;
constexpr int WGMMA_THREADS = 256;
constexpr int WARPGROUP_SIZE = 128;

#define WGMMA_SMEM_DESC_ENCODE(x) ((((uint64_t)(x)) & 0x3FFFF) >> 4)

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t wgmma_smem_desc(half *ptr) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0;
  desc |= WGMMA_SMEM_DESC_ENCODE(addr);
  desc |= WGMMA_SMEM_DESC_ENCODE(16ULL) << 16;
  desc |= WGMMA_SMEM_DESC_ENCODE(1024ULL) << 32;
  desc |= 1ULL << 62;
  return desc;
}

#define WGMMA_M64N128K16_F16_BODY(SCALE_D)                                     \
  asm volatile(                                                                 \
      "{\n"                                                                   \
      "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "                  \
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "                             \
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "                             \
      " %16, %17, %18, %19, %20, %21, %22, %23, "                             \
      " %24, %25, %26, %27, %28, %29, %30, %31},"                             \
      " %32, %33, " SCALE_D ", 1, 1, 0, 0;\n"                                 \
      "}\n"                                                                   \
      : "+r"(d[0][0]), "+r"(d[0][1]), "+r"(d[0][2]), "+r"(d[0][3]),           \
        "+r"(d[1][0]), "+r"(d[1][1]), "+r"(d[1][2]), "+r"(d[1][3]),           \
        "+r"(d[2][0]), "+r"(d[2][1]), "+r"(d[2][2]), "+r"(d[2][3]),           \
        "+r"(d[3][0]), "+r"(d[3][1]), "+r"(d[3][2]), "+r"(d[3][3]),           \
        "+r"(d[4][0]), "+r"(d[4][1]), "+r"(d[4][2]), "+r"(d[4][3]),           \
        "+r"(d[5][0]), "+r"(d[5][1]), "+r"(d[5][2]), "+r"(d[5][3]),           \
        "+r"(d[6][0]), "+r"(d[6][1]), "+r"(d[6][2]), "+r"(d[6][3]),           \
        "+r"(d[7][0]), "+r"(d[7][1]), "+r"(d[7][2]), "+r"(d[7][3])            \
      : "l"(desc_a), "l"(desc_b))

__device__ __forceinline__ void wgmma_m64n128k16_f16(half *sA, half *sB,
                                                       uint32_t d[8][4],
                                                       int scale_d) {
  const uint64_t desc_a = wgmma_smem_desc(sA);
  const uint64_t desc_b = wgmma_smem_desc(sB);
  if (scale_d == 0)
    WGMMA_M64N128K16_F16_BODY("0");
  else
    WGMMA_M64N128K16_F16_BODY("1");
}

// TN tensor map: inner dim = BlockMinor (K), outer = BlockMajor (M or N).
inline CUtensorMap encode_tma_tn(void *ptr, int blocks_outer, int blocks_k,
                                   int block_major, int block_minor) {
  CUtensorMap map{};
  const uint64_t global_dim[2] = {
      static_cast<uint64_t>(block_minor) * blocks_k,
      static_cast<uint64_t>(block_major) * blocks_outer};
  const uint64_t global_stride[1] = {
      static_cast<uint64_t>(block_minor) * blocks_k * sizeof(half)};
  const uint32_t box_dim[2] = {static_cast<uint32_t>(block_minor),
                               static_cast<uint32_t>(block_major)};
  const uint32_t elem_stride[2] = {1, 1};
  const CUresult err = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, ptr, global_dim, global_stride,
      box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (err != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(err, &msg);
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n",
                 msg ? msg : "unknown");
    std::exit(EXIT_FAILURE);
  }
  return map;
}

inline CUtensorMap encode_wgmma_tma_a(const half *A, int M, int K) {
  return encode_tma_tn(const_cast<half *>(A), M / BM, K / WGMMA_BK, BM,
                       WGMMA_BK);
}

// B is NxK row-major (K-major / TN layout).
inline CUtensorMap encode_wgmma_tma_b(const half *B_nk, int N, int K) {
  return encode_tma_tn(const_cast<half *>(B_nk), N / BN, K / WGMMA_BK, BN,
                       WGMMA_BK);
}

// B[K,N] row-major -> B_t[N,K] row-major (K contiguous) for TN WGMMA.
inline void transpose_b_kn_to_nk(const std::vector<half> &B_kn,
                                 std::vector<half> &B_nk, int N, int K) {
  B_nk.resize(B_kn.size());
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      B_nk[static_cast<size_t>(n) * K + k] =
          B_kn[static_cast<size_t>(k) * N + n];
}

template <int STAGES, typename Kernel>
inline void wgmma_request_smem(Kernel kernel) {
  constexpr int smem_bytes =
      STAGES * (BM * WGMMA_BK + WGMMA_BK * BN) * static_cast<int>(sizeof(half));
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
}

template <int STAGES>
__global__ void __launch_bounds__(WGMMA_THREADS)
    gemm_tma_wgmma(__grid_constant__ const CUtensorMap tensor_a,
                   __grid_constant__ const CUtensorMap tensor_b,
                   float *__restrict__ C, int M, int N, int K) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int A_STAGE = BM * WGMMA_BK;
  constexpr int B_STAGE = WGMMA_BK * BN;
  constexpr uint32_t TX_BYTES = (A_STAGE + B_STAGE) * sizeof(half);

  __shared__ alignas(128) half As[STAGES][A_STAGE];
  __shared__ alignas(128) half Bs[STAGES][B_STAGE];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> full[STAGES];
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[STAGES];

  const int wg = threadIdx.x / WARPGROUP_SIZE;
  const int lane = threadIdx.x % WARPGROUP_SIZE;
  const int nK = K / WGMMA_BK;

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      init(&full[s], WARPGROUP_SIZE + 1);
      init(&empty[s], WARPGROUP_SIZE + 1);
    }
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }
  __syncthreads();

  if (wg == 0) {
    if (lane == 0) {
      int q = 0;
      for (int kt = 0; kt < nK; ++kt, ++q) {
        if (q == STAGES)
          q = 0;
        empty[q].wait(empty[q].arrive());
        const int32_t ca[2] = {kt * WGMMA_BK,
                               static_cast<int32_t>(blockIdx.y * BM)};
        const int32_t cb[2] = {kt * WGMMA_BK,
                               static_cast<int32_t>(blockIdx.x * BN)};
        auto *bar = reinterpret_cast<uint64_t *>(
            cuda::device::barrier_native_handle(full[q]));
        cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster,
                                        cuda::ptx::space_global, As[q],
                                        &tensor_a, ca, bar);
        cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster,
                                        cuda::ptx::space_global, Bs[q],
                                        &tensor_b, cb, bar);
        (void)cuda::device::barrier_arrive_tx(full[q], 1, TX_BYTES);
      }
    }
    return;
  }

  for (int s = 0; s < STAGES; ++s)
    (void)empty[s].arrive();

  uint32_t acc[2][8][4];
  memset(acc, 0, sizeof(acc));
  int q = 0;
  for (int kt = 0; kt < nK; ++kt, ++q) {
    if (q == STAGES)
      q = 0;
    full[q].wait(full[q].arrive());
    wgmma_fence();
#pragma unroll
    for (int mi = 0; mi < BM / 64; ++mi) {
      half *tile_a = As[q] + mi * WGMMA_BK * 64;
#pragma unroll
      for (int kk = 0; kk < WGMMA_BK / 16; ++kk) {
        wgmma_m64n128k16_f16(tile_a + kk * 16, Bs[q] + kk * 16, acc[mi], 1);
      }
    }
    wgmma_commit_group();
    wgmma_wait_group();
    (void)empty[q].arrive();
  }

  const int warp = lane / 32;
  const int lane32 = lane % 32;
  const int row = warp * 16 + lane32 / 4;
  float *block_c = C + blockIdx.y * BM * N + blockIdx.x * BN;
#pragma unroll
  for (int mi = 0; mi < BM / 64; ++mi) {
    const int yo = mi * 64;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const int col = g * 16 + 2 * (lane32 % 4);
      half h[8];
      *reinterpret_cast<uint32_t *>(&h[0]) = acc[mi][g][0];
      *reinterpret_cast<uint32_t *>(&h[2]) = acc[mi][g][1];
      *reinterpret_cast<uint32_t *>(&h[4]) = acc[mi][g][2];
      *reinterpret_cast<uint32_t *>(&h[6]) = acc[mi][g][3];
      block_c[(row + yo) * N + col] = __half2float(h[0]);
      block_c[(row + yo) * N + col + 1] = __half2float(h[1]);
      block_c[(row + yo + 8) * N + col] = __half2float(h[2]);
      block_c[(row + yo + 8) * N + col + 1] = __half2float(h[3]);
      block_c[(row + yo) * N + col + 8] = __half2float(h[4]);
      block_c[(row + yo) * N + col + 9] = __half2float(h[5]);
      block_c[(row + yo + 8) * N + col + 8] = __half2float(h[6]);
      block_c[(row + yo + 8) * N + col + 9] = __half2float(h[7]);
    }
  }
#endif
}
