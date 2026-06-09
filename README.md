# Pipelining a Tensor-Core GEMM with `cp.async`

A CUDA experiment showing the benefit of software pipelining
(`cp.async` / `cuda::pipeline`).

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
shared loads with `cp.async`.


---

## The kernel

`C = A · B`, with `A` (M×K) and `B` (K×N) in `half`, `C` (M×N) in `float`.

- **Block tile** `BM×BN = 128×128`, K streamed in `BK = 16` chunks.
- **256 threads/block (8 warps)**, arranged `4×2`; each warp owns a `32×64`
  output sub-tile computed with `16×16×16` `wmma` MMA fragments.
- A/B tiles are staged through shared memory with 128-bit (`float4`) loads.

Three variants differ **only** in how the next K-tile is staged
(`src/kernels.cuh`):

| Variant | Staging |
| --- | --- |
| **`sync`** | synchronous shared memory (`__syncthreads`) — load *then* compute |
| **`cp.async`** | raw `__pipeline_memcpy_async`, multi-stage prefetch |
| **`cuda::pipeline`** | the multi-stage producer/consumer abstraction |

---

## Results (H100)

```
N=4096, stages=3
kernel                time (ms)      TFLOP/s      speedup  max rel err
------                ---------      -------      -------  -----------
sync                      2.042         67.3        1.00x     0.00e+00
cp.async                  1.586         86.7        1.29x     0.00e+00
cuda::pipeline            1.600         85.9        1.28x     0.00e+00

N=8192, stages=3
sync                     16.446         66.9        1.00x     0.00e+00
cp.async                 12.558         87.6        1.31x     0.00e+00
cuda::pipeline           12.639         87.0        1.30x     0.00e+00
```

**`cp.async` prefetching is a clean ~1.28–1.31x over the synchronous baseline**,
in a real GEMM at normal occupancy. That is the headline: *pipelining wins here.*

Correctness: every variant is checked against the synchronous kernel
(`max rel err = 0`), which is itself cross-checked against a CPU reference at
small `N`.

### Caveats

- **Absolute TFLOP/s is below cuBLAS** (~300+ TFLOP/s FP16 on H100). This is a
  teaching kernel that isolates *the pipelining effect*. It omits register
  double-buffering, shared-memory swizzling and epilogue tricks a production
  GEMM uses. The **relative** `sync` vs `cp.async` comparison is the point.


---

## Building and Running

Requires a CUDA toolkit (tested with CUDA 13.3) and an sm_80+ GPU (tensor cores
+ `cp.async`; tested on an H100). The Makefile auto-detects the target
architecture from the first visible GPU (`sm_80` for A100, `sm_90` for H100).

```bash
make                       # build -> build/gemm_pipeline
make run                   # N=4096, stages=3
make run N=8192 STAGES=2   # override size / pipeline depth

./build/gemm_pipeline --help
```
