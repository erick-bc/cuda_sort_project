NVCC ?= /usr/local/cuda-12.6/bin/nvcc

sort: benchmark.cu template.cu
	$(NVCC) $^ -o $@.exe

clean:
	rm -f a.out sort

# do not comment this out
# submit:
# 	zip $(USER)_project.zip Makefile *.cu *.cuh *.hpp *.cpp
# 	cp $(USER)_project.zip insert_directory
