This is a CUDA project that will implement merge sort and bitonic sort.

You need an Nvidia GPU to run this code that is capable of running CUDA 12.6.
Look up instructions online for compatibility if needed.

# Running the code

On the terminal, run `make`. The output executable is `out.exe` and can be run as
`./out.exe`. 
If modified, you can run `make clean` and then `make` to see the new results.

# Running the code (after MAKEFILE CHANGES)
 1. Run 'make clean' 
 2. compile 'nvcc -O3 -use_fast_math -arch=sm_86 -Xcompiler -O3 benchmark.cu template.cu -o sort.exe'
 3. ./sort.exe
