#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BS 1024

// Credit for the stand in kernels goes to
// https://github.com/rbga/CUDA-Merge-and-Bitonic-Sort/tree/master
// but these are NOT the final kernels.
// We have to optimize using various techniques.
__device__ void merge(int* arr, int* temp, int left, int middle, int right) {
    int i = left;
    int j = middle;
    int k = left;

    while (i < middle && j < right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        }
        else {
            temp[k++] = arr[j++];
        }
    }

    while (i < middle) {
        temp[k++] = arr[i++];
    }
    while (j < right) {
        temp[k++] = arr[j++];
    }

    for (int x = left; x < right; x++) {
        arr[x] = temp[x];
    }
}

// KERNEL 1
// GPU Kernel for Merge Sort
__global__ void merge_sort_gpu(int* arr, int* temp, int n, int width) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int left   = tid * width;
    int middle = min(left + width / 2, n);
    int right  = min(left + width, n);

    if (left < n && middle < n) {
        merge(arr, temp, left, middle, right);
    }
}

// KERNEL 2
// GPU Kernel Implementation of Bitonic Sort
__global__ void bitonic_sort_gpu(int* arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    unsigned int ij = i ^ j;
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
    const int GS = (size + BS - 1) / BS;

    // Copy input into the output buffer; kernels sort C in-place
    cudaMemcpy(C, A, size * sizeof(int), cudaMemcpyDeviceToDevice);

    if (kernel_id == 1) {
        // merge sort
        int *gpu_temp;
        cudaMalloc(&gpu_temp, size * sizeof(int));
        for (int wid = 1; wid < size; wid *= 2) {
            merge_sort_gpu<<<GS, BS>>>(C, gpu_temp, size, wid * 2);
        }
        cudaFree(gpu_temp);
    }
    else if (kernel_id == 2) {
        // bitonic sort
        for (int k = 2; k <= size; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonic_sort_gpu<<<GS, BS>>>(C, j, k);
            }
        }
    }
    else {
        // default: merge sort
        int *gpu_temp;
        cudaMalloc(&gpu_temp, size * sizeof(int));
        for (int wid = 1; wid < size; wid *= 2) {
            merge_sort_gpu<<<GS, BS>>>(C, gpu_temp, size, wid * 2);
        }
        cudaFree(gpu_temp);
    }
    // Make sure the kernel has finished.
    cudaDeviceSynchronize();
}
