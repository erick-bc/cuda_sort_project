# ==============================================================================
# CUDA Parallel Sort Project - University of Houston
# Target Hardware: NVIDIA RTX 3080 (Ampere sm_86)
# ==============================================================================

NVCC := /usr/local/cuda-12.6/bin/nvcc

# Compiler Flags - Optimized for Vanguard 12.5
# --std=c++17: Modern CUDA features
# -O3: Maximum compiler optimization
# -use_fast_math: Hardware-level math units
# -arch=sm_86: RTX 3080 Ampere architecture
# -lineinfo: Source code mapping for debugging
# -Xptxas -O3: PTX assembly-level optimization
# -Xptxas -v: Verbose output to check register usage and memory spills!
NVCC_FLAGS := --std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo \
              -Xptxas -O3 -Xptxas -v \
              -Xcompiler -O3 -Xcompiler -Wall

TARGET := sort.exe
SOURCES := benchmark.cu template.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean run