#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <string>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                      \
        cudaError_t err__ = (expr);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n", __FILE__, \
                         __LINE__, #expr, cudaGetErrorString(err__));        \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------
inline void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// Largest relative error between two device-computed results (already on host).
inline double max_rel_error(const std::vector<float>& ref,
                            const std::vector<float>& test) {
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = ref[i];
        double t = test[i];
        double denom = std::max(1e-3, std::fabs(r));
        worst = std::max(worst, std::fabs(r - t) / denom);
    }
    return worst;
}

// ---------------------------------------------------------------------------
// GPU timing via CUDA events. Returns average milliseconds per launch.
// ---------------------------------------------------------------------------
template <typename LaunchFn>
double time_kernel_ms(LaunchFn&& launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

// GFLOP/s for an N x N x N GEMM (2*N^3 flops) given per-launch time in ms.
inline double gemm_gflops(int n, double ms) {
    double flops = 2.0 * static_cast<double>(n) * n * n;
    return flops / (ms * 1.0e6);
}
