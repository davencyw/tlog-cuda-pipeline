// Benchmark harness for the realistic CUDA pipeline experiment:
// FP16 tiled GEMM on tensor cores.
//
// We compare:
//   (A) sync           - synchronous shared-memory staging
//   (B) cp.async       - raw asynchronous copy intrinsics, multi-stage
//   (C) cuda::pipeline - multi-stage software pipeline abstraction
//
// All kernels are verified against a CPU reference (small N) and the baseline.

#include "common.cuh"
#include "kernels.cuh"

#include <cstring>
#include <cuda_fp16.h>

namespace {

struct Config {
  int n = 4096;   // square M=N=K, multiple of max(BM,BN,BK)
  int stages = 3; // pipeline depth for cp.async / cuda::pipeline
  int iters = 100;
  int warmup = 30;
  unsigned seed = 1234;
};

void print_usage(const char *prog) {
  std::printf("Usage: %s [options]\n"
              "  --n <int>       square dimension M=N=K, multiple of %d "
              "(default 4096)\n"
              "  --stages <int>  pipeline depth, 2-4 (default 3)\n"
              "  --iters <int>   timed iterations (default 100)\n"
              "  --warmup <int>  warmup iterations (default 30)\n"
              "  --seed <int>    RNG seed (default 1234)\n"
              "  -h, --help      show this help\n",
              prog, BM);
}

bool parse_args(int argc, char **argv, Config &cfg) {
  for (int i = 1; i < argc; ++i) {
    auto next = [&](int &out) {
      if (i + 1 >= argc)
        return false;
      out = std::atoi(argv[++i]);
      return true;
    };
    if (!std::strcmp(argv[i], "--n")) {
      if (!next(cfg.n))
        return false;
    } else if (!std::strcmp(argv[i], "--stages")) {
      if (!next(cfg.stages))
        return false;
    } else if (!std::strcmp(argv[i], "--iters")) {
      if (!next(cfg.iters))
        return false;
    } else if (!std::strcmp(argv[i], "--warmup")) {
      if (!next(cfg.warmup))
        return false;
    } else if (!std::strcmp(argv[i], "--seed")) {
      int s = 0;
      if (!next(s))
        return false;
      cfg.seed = static_cast<unsigned>(s);
    } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return false;
    }
  }
  return true;
}

struct Result {
  const char *name;
  double ms;
  double tflops;
  double rel_err;
};

double gemm_tflops(int n, double ms) {
  double flops = 2.0 * static_cast<double>(n) * n * n;
  return flops / (ms * 1.0e9);
}

// CPU reference for a small N sanity check (FP32 accumulate of half inputs).
void cpu_gemm(const std::vector<half> &A, const std::vector<half> &B,
              std::vector<float> &C, int n) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < n; ++k)
        acc += __half2float(A[i * n + k]) * __half2float(B[k * n + j]);
      C[i * n + j] = acc;
    }
}

} // namespace

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (cfg.n % BM != 0 || cfg.n % BN != 0 || cfg.n % BK != 0) {
    std::fprintf(stderr, "N (%d) must be a multiple of %d\n", cfg.n, BM);
    return EXIT_FAILURE;
  }
  if (cfg.stages < 2 || cfg.stages > 4) {
    std::fprintf(stderr, "--stages must be in [2, 4]\n");
    return EXIT_FAILURE;
  }

  const int N = cfg.n;
  const size_t elems = static_cast<size_t>(N) * N;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("Device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);
  std::printf(
      "GEMM: %d x %d x %d (FP16 in, FP32 acc), tile %dx%dx%d, stages %d\n", N,
      N, N, BM, BN, BK, cfg.stages);
  std::printf("Iters: %d (warmup %d)\n\n", cfg.iters, cfg.warmup);

  // Host data (half inputs, small magnitude to keep FP16 accumulation sane).
  std::vector<float> hAf(elems), hBf(elems);
  fill_random(hAf, cfg.seed);
  fill_random(hBf, cfg.seed + 1);
  std::vector<half> hA(elems), hB(elems);
  for (size_t i = 0; i < elems; ++i) {
    hA[i] = __float2half(0.1f * hAf[i]);
    hB[i] = __float2half(0.1f * hBf[i]);
  }

  half *dA = nullptr, *dB = nullptr;
  float *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, elems * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dB, elems * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dC, elems * sizeof(float)));
  CUDA_CHECK(
      cudaMemcpy(dA, hA.data(), elems * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(dB, hB.data(), elems * sizeof(half), cudaMemcpyHostToDevice));

  const dim3 block(THREADS);
  const dim3 grid(N / BN, N / BM);

  auto download = [&](std::vector<float> &dst) {
    dst.resize(elems);
    CUDA_CHECK(cudaMemcpy(dst.data(), dC, elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
  };

  // Reference = sync kernel output. Cross-checked against CPU for small N.
  std::vector<float> reference;
  gemm_sync<<<grid, block>>>(dA, dB, dC, N, N, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  download(reference);

  if (N <= 512) {
    std::vector<float> cpu(elems);
    cpu_gemm(hA, hB, cpu, N);
    std::printf("CPU cross-check (N=%d): max rel err vs sync kernel = %.2e\n\n",
                N, max_rel_error(cpu, reference));
  }

  auto bench = [&](const char *name, auto &&launch) -> Result {
    double ms = time_kernel_ms(launch, cfg.warmup, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> out;
    download(out);
    return {name, ms, gemm_tflops(N, ms), max_rel_error(reference, out)};
  };

  auto launch_staged = [&](auto ksel) {
    switch (cfg.stages) {
    case 2:
      ksel(std::integral_constant<int, 2>{});
      break;
    case 3:
      ksel(std::integral_constant<int, 3>{});
      break;
    case 4:
      ksel(std::integral_constant<int, 4>{});
      break;
    }
  };

  std::vector<Result> results;
  results.push_back(
      bench("sync", [&] { gemm_sync<<<grid, block>>>(dA, dB, dC, N, N, N); }));
  results.push_back(bench("cp.async", [&] {
    launch_staged([&](auto S) {
      gemm_cpasync<S.value><<<grid, block>>>(dA, dB, dC, N, N, N);
    });
  }));
  results.push_back(bench("cuda::pipeline", [&] {
    launch_staged([&](auto S) {
      gemm_pipeline<S.value><<<grid, block>>>(dA, dB, dC, N, N, N);
    });
  }));

  std::printf("%-18s %12s %12s %12s %12s\n", "kernel", "time (ms)", "TFLOP/s",
              "speedup", "max rel err");
  std::printf("%-18s %12s %12s %12s %12s\n", "------", "---------", "-------",
              "-------", "-----------");
  const double base_ms = results.front().ms;
  for (const auto &r : results)
    std::printf("%-18s %12.3f %12.1f %11.2fx %12.2e\n", r.name, r.ms, r.tflops,
                base_ms / r.ms, r.rel_err);

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return EXIT_SUCCESS;
}
