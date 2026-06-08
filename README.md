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

## Results (A100)

```
N=4096, stages=3
kernel                time (ms)      TFLOP/s      speedup  max rel err
------                ---------      -------      -------  -----------
sync                      3.241         42.4        1.00x     0.00e+00
cp.async                  2.653         51.8        1.22x     0.00e+00
cuda::pipeline            3.014         45.6        1.08x     0.00e+00

N=8192, stages=3
sync                     24.869         44.2        1.00x     0.00e+00
cp.async                 19.950         55.1        1.25x     0.00e+00
cuda::pipeline           22.631         48.6        1.10x     0.00e+00
```

**`cp.async` prefetching is a clean ~1.22–1.25x over the synchronous baseline**,
in a real GEMM at normal occupancy. That is the headline: *pipelining wins here.*

Correctness: every variant is checked against the synchronous kernel
(`max rel err = 0`), which is itself cross-checked against a CPU reference at
small `N`.

### Caveats

- **Absolute TFLOP/s is below cuBLAS** (~150+ TFLOP/s FP16 on A100). This is a
  teaching kernel that isolates *the pipelining effect*. It omits register
  double-buffering, shared-memory swizzling and epilogue tricks a production
  GEMM uses. The **relative** `sync` vs `cp.async` comparison is the point.


---

## Building and Running

Requires a CUDA toolkit (tested with CUDA 13.0) and an sm_80+ GPU (tensor cores
+ `cp.async`; tested on an A100).

```bash
make                       # build -> build/gemm_pipeline (ARCH=sm_80 default)
make run                   # N=4096, stages=3
make run N=8192 STAGES=2   # override size / pipeline depth

./build/gemm_pipeline --help
```
