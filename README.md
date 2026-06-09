# Pipelining a Tensor-Core GEMM with `cp.async`

A CUDA experiment showing the benefit of software pipelining
(`cp.async` / `cuda::pipeline`), and how the same idea extends to Hopper
(TMA + WGMMA).

---

## Why

A software pipeline hides **memory latency** by overlapping the copy of the next
tile with compute on the current one. It only helps when latency is *exposed*,
i.e. the warp scheduler isn't already hiding it for free.

That almost never happens for plain memory streaming at full occupancy (you just
become bandwidth-bound). It **does** happen for **tensor-core GEMM**: the tensor
cores finish a tile's matrix-multiply so fast that staging the next A/B tiles
from global memory becomes the bottleneck. Even at high occupancy and near-peak
hardware utilization. This is exactly why cuBLAS/CUTLASS pipeline their global-
shared loads with `cp.async` (and, on Hopper, TMA bulk copies).

---

## The kernel

`C = A · B`, with `A` (M×K) and `B` (K×N) in `half`, `C` (M×N) in `float`.

- **Block tile** `BM×BN = 128×128`, K streamed in `BK = 16` chunks (WMMA path).
- **256 threads/block (8 warps)**, arranged `4×2`; each warp owns a `32×64`
  output sub-tile computed with `16×16×16` `wmma` MMA fragments.
- A/B tiles are staged through shared memory with 128-bit (`float4`) loads.

Variants differ in how the next K-tile is staged and computed:

| Variant | Staging / compute | File |
| --- | --- | --- |
| **`sync`** | synchronous shared memory — load *then* compute | `kernels.cuh` |
| **`cp.async`** | raw `__pipeline_memcpy_async`, multi-stage prefetch | `kernels.cuh` |
| **`cuda::pipeline`** | multi-stage producer/consumer abstraction | `kernels.cuh` |
| **`TMA`** | Hopper `cp.async.bulk.tensor` + mbarrier, WMMA compute | `tma.cuh` |
| **`WGMMA no pipe`** | 128B-swizzled TMA + warpgroup MMA, single-stage (load then compute) | `wgmma.cuh` |
| **`TMA+WGMMA`** | same tiled WGMMA, multi-stage TMA/WGMMA pipeline | `wgmma.cuh` |

The WMMA variants (`sync` / `cp.async` / `cuda::pipeline` / `TMA`) use
`BK = 16`. The WGMMA path uses `BK = 64` (required for 128B swizzle), TN
layout (`A` row-major M×K, `B` transposed to N×K), and warp specialization:
warpgroup 0 issues TMA loads, warpgroup 1 runs `m64n128k16` WGMMA.

---

## Results (H100)

```
N=4096, stages=3
kernel                time (ms)      TFLOP/s      speedup  max rel err
------                ---------      -------      -------  -----------
sync                      2.098         65.5        1.00x     0.00e+00
cp.async                  1.637         83.9        1.28x     0.00e+00
cuda::pipeline            1.643         83.6        1.28x     0.00e+00
TMA                       1.668         82.4        1.26x     0.00e+00
WGMMA no pipe             0.375        366.2        5.59x     2.57e+00
TMA+WGMMA                 0.278        494.5        7.55x     2.57e+00

N=8192, stages=3
sync                     16.872         65.2        1.00x     0.00e+00
cp.async                 12.970         84.8        1.30x     0.00e+00
cuda::pipeline           12.915         85.1        1.31x     0.00e+00
TMA                      13.201         83.3        1.28x     0.00e+00
WGMMA no pipe             2.287        480.8        7.38x     5.29e+00
TMA+WGMMA                 2.289        480.3        7.37x     5.29e+00
```

**WMMA pipelining** (`cp.async` / `cuda::pipeline` / `TMA`) is a clean
**~1.28–1.31×** over the synchronous baseline — the original headline.

**WGMMA** is a different regime: warpgroup tensor cores plus 128B-swizzled TMA
staging are far faster than block-scoped WMMA, even without pipelining
(~366 TFLOP/s at N=4096). At N=4096, a 3-stage TMA/WGMMA pipeline adds another
**~1.35×** over the single-stage path (~495 vs ~366 TFLOP/s). At N=8192 the
pipelined and non-pipelined WGMMA paths converge (~480 TFLOP/s).

Correctness: WMMA variants are checked against the synchronous kernel
(`max rel err = 0`), which is cross-checked against a CPU reference at small
`N`. WGMMA uses **f16 accumulators**; the reported relative error vs the f32
WMMA reference reflects that (max absolute error ~0.7 at N=4096).

### Caveats

- **Absolute TFLOP/s is below cuBLAS** for the WMMA teaching kernels (~80–85
  TFLOP/s). They isolate the *pipelining effect* and omit register
  double-buffering, swizzling, and epilogue tricks a production GEMM uses.
- **WGMMA requires Hopper** (`sm_90+`) and is built with `compute_90a` (plain
  `sm_90` is insufficient for WGMMA/TMA accelerated ops). `N` must be a multiple
  of 64 for the WGMMA path.
- WGMMA benchmarks transpose `B` to TN layout on the host before the timed run.

---

## Building and Running

Requires a CUDA toolkit (tested with CUDA 13.3) and an sm_80+ GPU (tensor cores
+ `cp.async`; tested on an H100). The Makefile auto-detects the target
architecture from the first visible GPU (`sm_80` for A100, `sm_90a` for H100).

```bash
make                       # build -> build/gemm_pipeline
make run                   # N=4096, stages=3
make run N=8192 STAGES=2   # override size / pipeline depth

./build/gemm_pipeline --help
```

On Hopper, the harness also runs `TMA`, `WGMMA no pipe`, and `TMA+WGMMA`
when `N % 64 == 0`. `--stages` controls the WMMA/TMA pipeline depth and the
pipelined WGMMA variant; `WGMMA no pipe` is always single-stage.
