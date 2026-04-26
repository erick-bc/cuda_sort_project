#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <algorithm>// For std::min and std::max
#include <climits> // Added for INT_MAX
#define BS 128 // more registers per thread
#define TILE 1024
#define BANK_PADDING 8 


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
using std::size_t;

// Credit for the code setup goes to 
// Dr. Wu, replaced with code suited to our tasks.
// Credit for the stand in kernels goes to
// https://github.com/rbga/CUDA-Merge-and-Bitonic-Sort/tree/master
// but these are NOT the final kernels.
// We have to optimize using various techniques.

// - - Helper Functions (Professor's Logic: Co-rank and sequential merge) - - 
// --- 1. Vectorized search ---
// Using binary search but unrolling the comparison logic
///
__device__ __forceinline__ int co_rank_safe(int k, const int* __restrict__ A, int a_len, const int* __restrict__ B, int b_len) {
    int low = (k > b_len) ? k - b_len : 0;
    int high = (k < a_len) ? k : a_len;
    // unroll binary search
    #pragma unroll 10
    while (low < high) {
        int i = low + (high - low) / 2;
        int j = k - 1 - i;
        if (A[i] < B[j]) low = i + 1;
        else high = i;
    }
    return low;
}
// Sequential merge with ILP padding
// Unrolled sequential merge
///
__device__ void merge_sequential(const int* A, int m, const int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    #pragma unroll 8
    while (i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else C[k++] = B[j++];
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}
// --- Vertorized block sort (Phase 1 turbo) ---
///
// 
__global__ void block_sort_kernel(int* __restrict__ in, int* __restrict__ out, int size) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int block_start = blockIdx.x * TILE;

    if (block_start + TILE <= size) {
        int4* in4 = (int4*)(in + block_start);
        int4* s4  = (int4*)s_data;
        for (int i = tid; i < TILE/4; i += blockDim.x) s4[i] = in4[i];
    } else {
        for (int i = tid; i < TILE; i += blockDim.x)
            s_data[i] = (block_start + i < size) ? in[block_start + i] : INT_MAX;
    }
    __syncthreads();

    for (int k = 2; k <= TILE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < TILE; i += blockDim.x) {
                unsigned int ij = i ^ j;
                if (ij > i) {
                    bool ascending = ((i & k) == 0);
                    if ((s_data[i] > s_data[ij]) == ascending) {
                        int tmp = s_data[i]; s_data[i] = s_data[ij]; s_data[ij] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (block_start + TILE <= size) {
        int4* out4 = (int4*)(out + block_start);
        int4* s4   = (int4*)s_data;
        for (int i = tid; i < TILE/4; i += blockDim.x) out4[i] = s4[i];
    } else {
        for (int i = tid; i < TILE; i += blockDim.x)
            if (block_start + i < size) out[block_start + i] = s_data[i];
    }
}
// - - Kernel 1, Coarsened Parallel Merge Kernel - -
// GPU Kernel for Merge Sort
template<int TILE_SIZE>
__global__ void __launch_bounds__(BS) merge_stage_kernel_v12(const int* __restrict__ in, int* __restrict__ out, int size, int wid) {
    extern __shared__ int shared_mem[];
    int* sA = shared_mem;
    int* sB = shared_mem + TILE_SIZE + BANK_PADDING; 
    __shared__ int splits[2];

    int tile_start_global = blockIdx.x * TILE_SIZE;
    if (tile_start_global >= size) return;

    int pair_start = (tile_start_global / wid) * wid;
    int seg_len = wid >> 1;
    int a_off = pair_start, a_len = (a_off + seg_len <= size) ? seg_len : max(0, size - a_off);
    int b_off = pair_start + seg_len, b_len = (b_off < size) ? min(seg_len, size - b_off) : 0;

    if (b_len <= 0) {
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            if (tile_start_global + i < size) out[tile_start_global + i] = in[tile_start_global + i];
        }
        return;
    }

    if (threadIdx.x == 0) {
        int tile_rel = tile_start_global - pair_start;
        splits[0] = co_rank_safe(tile_rel, in + a_off, a_len, in + b_off, b_len);
        int end_rel = min(tile_rel + TILE_SIZE, a_len + b_len);
        splits[1] = co_rank_safe(end_rel, in + a_off, a_len, in + b_off, b_len);
    }
    __syncthreads();

    int a_s_r = splits[0], a_e_r = splits[1];
    int b_s_r = (tile_start_global - pair_start) - a_s_r;
    int tile_end_rel = min((tile_start_global - pair_start) + TILE_SIZE, a_len + b_len);
    int b_e_r = tile_end_rel - a_e_r;

    int a_load = a_e_r - a_s_r;
    int b_load = b_e_r - b_s_r;

    for (int i = threadIdx.x; i < a_load; i += blockDim.x) sA[i] = in[a_off + a_s_r + i];
    for (int i = threadIdx.x; i < b_load; i += blockDim.x) sB[i] = in[b_off + b_s_r + i];
    __syncthreads();

    int items_per_thread = TILE_SIZE / blockDim.x; 
    int t_start = threadIdx.x * items_per_thread;
    int t_end = min(t_start + items_per_thread, a_load + b_load);

    if (t_start < t_end) {
        int a_s = co_rank_safe(t_start, sA, a_load, sB, b_load);
        int a_e = co_rank_safe(t_end, sA, a_load, sB, b_load);
        int b_s = t_start - a_s;
        int b_e = t_end - a_e;
        
        int out_idx = tile_start_global + t_start;
        int write_len = t_end - t_start;
        
        // Check both start AND end bounds to prevent memory corruption
        if (out_idx >= 0 && out_idx + write_len <= size) {
            merge_sequential(sA + a_s, a_e - a_s, sB + b_s, b_e - b_s, out + out_idx);
        }
    }
}
// - - kernel 2: (bitonic kernel)
__global__ void bitonic_sort_gpu(int *arr, int j, int k, int n)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n)
        return;

    unsigned int ij = i ^ j;

    if (ij > i && ij < n)
    {
        if ((i & k) == 0)
        {
            if (arr[i] > arr[ij])
            {
                int tmp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = tmp;
            }
        }
        else
        {
            if (arr[i] < arr[ij])
            {
                int tmp = arr[i];
                arr[i] = arr[ij];
                arr[ij] = tmp;
            }
        }
    }
}

__global__ void padded_kernel(int *arr, int size, int padded) { // Kernel to pad the array to the next power of 2 with INT_MAX so that bitonic sort                                                           
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;     // does not have to worry about out of bounds access.
    if (i < padded) {
        arr[i] = (i < size) ? arr[i] : INT_MAX; // Pad temp array with INT_MAX for sorting
    }
}
//------------------------------------------------------------------------------
// (Feel free to change anything if needed)
// Kernel Launcher: Manages the execution of sorting algorithms on the GPU.
// 
extern "C" void launch_sort_kernel(int kernel_id, int *A, int *C, int size) {
    if (kernel_id == 1 || kernel_id == 0) {
        int *gpu_temp;
        gpuErrchk(cudaMalloc(&gpu_temp, size * sizeof(int)));
        
        block_sort_kernel<<<(size + 1023) / 1024, BS, 1024 * sizeof(int)>>>(A, C, size);
        gpuErrchk(cudaDeviceSynchronize());

        int *src = C, *dst = gpu_temp;
        for (int wid = 2048; wid <= size * 2; wid *= 2) {
            int num_tiles = (size + 1023) / 1024;
            size_t smem_size = (TILE + TILE + BANK_PADDING) * sizeof(int);
            merge_stage_kernel_v12<TILE><<<num_tiles, BS, smem_size>>>(src, dst, (int)size, wid);
            gpuErrchk(cudaDeviceSynchronize());
            int *tmp = src; src = dst; dst = tmp;
            if (wid >= size && size > 1) break;
        }

        if (src != C) {
            gpuErrchk(cudaMemcpy(C, src, size * sizeof(int), cudaMemcpyDeviceToDevice));
        }
        cudaFree(gpu_temp);

    } 
    else if (kernel_id == 2)
    {
        int padded = 1;                    // padded size to next power of 2 for bitonic sort
        while (padded < size) padded <<= 1;   // So that bitonic sort can handle non-power-of-2 sizes.

        int *gpu_temp;
        gpuErrchk(cudaMalloc(&gpu_temp, padded * sizeof(int)));
        gpuErrchk(cudaMemcpy(gpu_temp, A, size * sizeof(int), cudaMemcpyDeviceToDevice));

        const int local_BS = 256; // Adjusted to 256 to better utilize GPU resources and hide latency.

        int buffer = padded - size; // Number of padding elements needed

        if(buffer > 0) // Only launch the padding kernel if we actually need to pad the array, avoiding 0 error and unnecessary kernel launch overhead.
        {
            const int padded_GS = (buffer + local_BS - 1) / local_BS;
            padded_kernel<<<padded_GS, local_BS>>>(gpu_temp, size, padded); // kernel to padded the array with INT_MAX to make it a power of 2
            gpuErrchk(cudaPeekAtLastError());
        }
        
        const int GS = (padded + local_BS - 1) / local_BS;

        for (int k = 2; k <= padded; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                bitonic_sort_gpu<<<GS, local_BS>>>(gpu_temp, j, k, padded);
                gpuErrchk(cudaPeekAtLastError());
            }
        }
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy(C, gpu_temp, size * sizeof(int), cudaMemcpyDeviceToDevice)); // Copy only the sorted portion back to the array, ignoring the padded INT_MAX values.
        cudaFree(gpu_temp);
    }
}

