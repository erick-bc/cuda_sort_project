This is a CUDA project that will implement merge sort and bitonic sort.

You need an Nvidia GPU to run this code that is capable of running CUDA 12.6.
Look up instructions online for compatibility if needed.

# Running the code

The workflow is as follows:
1. On the terminal, run `make all` or simply `make`. This makes the output executable `sort.exe`.
2. To get the work throughput, run `make run` to see `MKeys/s`.
3. To get memory bandwidth results, run `make throughput` to see `GB/s` and `Peak bandwidth %`. This can take some time.

If modified, you can run `make clean` and then run as normal.