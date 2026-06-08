# CUDA pipeline GEMM experiment
#
#   make            build the benchmark
#   make run        build and run with defaults
#   make profile    build and capture an Nsight Compute report
#   make clean      remove build artifacts
#
# Override the target architecture for non-A100 GPUs, e.g.:
#   make ARCH=sm_86

NVCC    ?= nvcc
ARCH    ?= sm_80
STD     ?= c++17

BUILD   := build
TARGET  := $(BUILD)/gemm_pipeline
SRC     := src/main.cu
DEPS    := src/common.cuh src/kernels.cuh

NVCCFLAGS := -O3 -std=$(STD) -arch=$(ARCH) --expt-relaxed-constexpr -lineinfo

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
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@

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
