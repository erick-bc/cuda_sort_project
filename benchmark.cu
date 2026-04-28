#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_sort_kernel(int kernel_id, int *A, int *C, int size);

#ifdef __cplusplus
}
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
        exit(code);
      }
   }
}

// THIS BENCHMARKING CODE COMES FROM DR. WU'S ASSIGNMENT 2.
// Verification helper: compare kernel result to reference
// return 1 for fail to pass verification; 0 otherwise
int verify_result(const int* reference, const int* result, int size, const char* kernel_name) {
    int errors = 0;
    for (int i = 0; i < size && errors < 10; i++) {
        int ref_val = reference[i];
        int out_val = result[i];
        if (ref_val != out_val) {
            printf("%s error at [%d]: %d vs %d\n", kernel_name, i, out_val, ref_val);
            errors++;
        }
    }
    return errors > 0 ? 1 : 0;
}

using std::size_t;

int main() {
    // Array sizes to test
    gpuErrchk(cudaSetDevice(0));

    int sizes[] = {(int) pow(2, 15) /* 32,768 */, 
                   (int) pow(2, 20) /* 1,048,576 */,
                   (int) pow(2, 25) /* 33,554,432 */, 
                   (int) pow(2, 27) /* 134,217,728 */,
                   (int) pow(2, 28) /* 268,435,456 */};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int fails = 0;
    // Loop over each array size
    for (int s = 0; s < num_sizes; s++) {
        size_t size = sizes[s];
        printf("=====================================================================================\n");
        printf("Sorting for size: %zu\n", size);
        printf("-------------------------------------------------------------------------------------\n");
        printf("%-20s %15s %15s\n", "Kernel", "Time (ms)", "MKeys/s");
        printf("-------------------------------------------------------------------------------------\n");

        int *A_d, *C_d;
        int *A_h, *result_h, *ref_h;

        // Allocate host memory
        A_h      = (int *)malloc(size * sizeof(int));
        result_h = (int *)malloc(size * sizeof(int));
        ref_h    = (int *)malloc(size * sizeof(int));

        // Initialize array A with random integer values
        for (size_t i = 0; i < size; i++) {
            A_h[i] = rand() % 1000000; // one million
        }

        // Allocate device memory
        gpuErrchk(cudaMalloc(&A_d, size * sizeof(int)));
        gpuErrchk(cudaMalloc(&C_d, size * sizeof(int)));

        // Copy host data to device
        gpuErrchk(cudaMemcpy(A_d, A_h, size * sizeof(int), cudaMemcpyHostToDevice));

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float time;

        // Compute reference result using Thrust sort (Radix)
        cudaMemcpy(C_d, A_d, size * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaEventRecord(start);
        thrust::device_ptr<int> d_ptr(C_d);
        thrust::sort(d_ptr, d_ptr + size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        float mkeys_sec = (size / 1e6f) / (time * 1e-3f);
        printf("%-20s %15.2f %15.2f\n", "Thrust Sort (Radix)", time, mkeys_sec);

        // Compute reference result using Thrust sort (Merge)
        cudaMemcpy(C_d, A_d, size * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaEventRecord(start);
        thrust::device_ptr<int> d_ptr2(C_d);
        thrust::sort(d_ptr2, d_ptr2 + size, thrust::less<int>());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        mkeys_sec = (size / 1e6f) / (time * 1e-3f);
        printf("%-20s %15.2f %15.2f\n", "Thrust Sort (Merge)", time, mkeys_sec);

        // Copy reference result back to host
        cudaMemcpy(ref_h, C_d, size * sizeof(int), cudaMemcpyDeviceToHost);

        const char *kernel_names[3] = {"", "Merge", "Bitonic"};

        // Run student kernels (IDs 1-2)
    for (int kernel_to_run = 1; kernel_to_run <= 2; kernel_to_run++) {
        gpuErrchk(cudaMemset(C_d, 0, size * sizeof(int)));

        cudaEventRecord(start);
        launch_sort_kernel(kernel_to_run, A_d, C_d, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
    
        mkeys_sec = (size / 1e6f) / (time * 1e-3f);
        printf("%-20s %15.2f %15.2f\n", kernel_names[kernel_to_run], time, mkeys_sec);

        // Copy the result from device to host and verify correctness
        cudaMemcpy(result_h, C_d, size * sizeof(int), cudaMemcpyDeviceToHost);
        fails += verify_result(ref_h, result_h, size, kernel_names[kernel_to_run]);
    }

        printf("=====================================================================================\n\n");

        // Cleanup resources for this size
        free(A_h);
        free(result_h);
        free(ref_h);
        cudaFree(A_d);
        cudaFree(C_d);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    if (fails > 0) {
        exit(1);
    }
    return 0;
}
