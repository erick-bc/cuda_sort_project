# Path to the compiler
NVCC := /usr/local/cuda-12.6/bin/nvcc

# Compiler Flags
# -O3: Maximum optimization
# -use_fast_math: Speeds up float math (crucial for GB/s gains)
# -arch=sm_86: Specifically targets the Ampere architecture (RTX 3080)
# -lineinfo: Keeps track of code lines in the binary (crucial for debugging illegal memory access)
NVCC_FLAGS = -O3 -use_fast_math -arch=sm_86 -lineinfo -Xcompiler -O3

# Host Compiler Flags
# -Wall: Show all warnings
# -O3: Optimize the C++ side (benchmark.cu logic)
XFLAGS := -Xcompiler -Wall -Xcompiler -O3

# Build Targets
TARGET := sort.exe

all: $(TARGET)

$(TARGET): benchmark.cu template.cu
	$(NVCC) $(NVCCFLAGS) $(XFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)