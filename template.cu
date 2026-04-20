#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
        exit(code);
      }
   }
}
#define BS 1024

using std::size_t;

// Credit for the code setup goes to 
// Dr. Wu, replaced with code suited to our tasks.
// Credit for the stand in kernels goes to
// https://github.com/rbga/CUDA-Merge-and-Bitonic-Sort/tree/master
// but these are NOT the final kernels.
// We have to optimize using various techniques.

// The following merge code is horribly inefficient for a GPU.
__device__ void merge(int* input, int* out, int left, int middle, int right)
{
    size_t i = left;
    size_t j = middle;
    size_t k = left;

    while (i < middle && j < right) {
        if (input[i] <= input[j]) {
            out[k++] = input[i++];
        } else {
            out[k++] = input[j++];
        }
    }

    while (i < middle) {
        out[k++] = input[i++];
    }

    while (j < right) {
        out[k++] = input[j++];
    }
}

// KERNEL 1
// GPU Kernel for Merge Sort
__global__ void merge_sort_gpu(int *arr, int *out, int n, int width) {
    size_t tid = (size_t) (BS * blockIdx.x + threadIdx.x);
    size_t left   = tid * width;
    size_t middle = min(left + width / 2, (size_t) n);
    size_t right  = min(left + width, (size_t) n);

    if (left < n && middle < n && right <= n) {
        merge(arr, out, left, middle, right);
    }
}

// KERNEL 2
// GPU Kernel Implementation of Bitonic Sort
__global__ void bitonic_sort_gpu(int* arr, int j, int k, int n) {
    unsigned int i = BS * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    unsigned int ij = i ^ j;
    if (ij >= n) return;
    if (ij > i) {
        if ((i & k) == 0) {
            if (arr[i] > arr[ij]) {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
        else {
            if (arr[i] < arr[ij]) {
                int temp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = temp;
            }
        }
    }
}


//------------------------------------------------------------------------------
// Kernel launcher function
//
// This function selects which kernel to run based on kernel_id.
extern "C" void launch_sort_kernel(int kernel_id, int *A, int *C, int size) {
    // Copy input into the output buffer; kernels sort C in-place
    cudaMemcpy(C, A, size * sizeof(int), cudaMemcpyDeviceToDevice);

    // Feel free to change anything here.
    // The ping ponging came from Gemini to avoid
    // new memory allocation. 
    if (kernel_id == 1 || kernel_id == 0) {
        // merge sort
        int *gpu_temp;
        gpuErrchk(cudaMalloc(&gpu_temp, size * sizeof(int)));

        int *in = C;
        int *out = gpu_temp;
        for (int wid = 2; wid <= size; wid *= 2) {
            int num_merges = size / wid;
            int GS_dynamic = (num_merges + BS - 1) / BS;

            merge_sort_gpu<<<GS_dynamic, BS>>>(in, out, size, wid);
            gpuErrchk(cudaDeviceSynchronize());

            // swap
            int *tmp = in;
            in = out;
            out = tmp;
        }
        cudaFree(gpu_temp);
    }
    else if (kernel_id == 2) {
        // bitonic sort
        const int GS = (size + BS - 1) / BS;
        for (int k = 2; k <= size; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonic_sort_gpu<<<GS, BS>>>(C, j, k, size);
                gpuErrchk(cudaDeviceSynchronize()); 
            }
        }
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
