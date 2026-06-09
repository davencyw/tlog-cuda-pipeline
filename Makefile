# CUDA pipeline GEMM experiment
#
#   make            build the benchmark
#   make run        build and run with defaults
#   make profile    build and capture an Nsight Compute report
#   make clean      remove build artifacts
#
# Override the target architecture if needed, e.g.:
#   make ARCH=sm_80    # A100
#   make ARCH=sm_90    # H100
# When ARCH is unset, the first visible GPU's compute capability is used.

NVCC    ?= nvcc
ARCH    ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                   | head -1 | tr -d '.' | sed 's/^/sm_/')
ifeq ($(ARCH),sm_)
ARCH    := sm_80
endif
STD     ?= c++17

# Hopper WGMMA/TMA need compute_90a (plain sm_90 is insufficient).
ifeq ($(ARCH),sm_90)
GENCODE := -gencode arch=compute_90a,code=sm_90a
else
GENCODE := -gencode arch=compute_$(subst sm_,,$(ARCH)),code=$(ARCH)
endif

BUILD   := build
TARGET  := $(BUILD)/gemm_pipeline
SRC     := src/main.cu
DEPS    := src/common.cuh src/kernels.cuh src/tma.cuh src/wgmma.cuh

NVCCFLAGS := -O3 -std=$(STD) $(GENCODE) --expt-relaxed-constexpr -lineinfo
LDFLAGS   := -lcuda

# Default runtime arguments (override on the command line, e.g. `make run N=8192`).
N      ?= 4096
STAGES ?= 3
RUNARGS := --n $(N) --stages $(STAGES)

# Metrics that make the latency-hiding effect visible in Nsight Compute.
NCU_METRICS := \
	smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
	sm__warps_active.avg.pct_of_peak_sustained_active,\
	sm__inst_executed.avg.per_cycle_active,\
	dram__throughput.avg.pct_of_peak_sustained_elapsed

.PHONY: all run profile clean

all: $(TARGET)

$(TARGET): $(SRC) $(DEPS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ $(LDFLAGS)

$(BUILD):
	mkdir -p $(BUILD)

run: $(TARGET)
	./$(TARGET) $(RUNARGS)

# Capture a profiler report (requires Nsight Compute `ncu`).
profile: $(TARGET)
	ncu --set full --metrics $(NCU_METRICS) \
		-o $(BUILD)/report ./$(TARGET) --n 2048 --stages $(STAGES) \
		--iters 1 --warmup 0

clean:
	rm -rf $(BUILD)
