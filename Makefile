NVCC ?= /usr/local/cuda-12.6/bin/nvcc

sort: benchmark.cu template.cu
	$(NVCC) $^ -o $@.exe -Xcompiler -Wall

clean:
	rm -f sort.exe