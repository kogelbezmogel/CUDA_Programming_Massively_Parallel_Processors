#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 4

__global__ void scan_kogge_stone_domino_style(float *A, float *B, int N, int *block_counter, int *flag, float *inter_values);


int main() {
    float *A_h, *A_d, *B_h, *B_d, *inter_values;
    int *flag, *block_counter;
    int N;

    N = 12;

    A_h = new float[N];
    B_h = new float[N];

    cudaMalloc((void**) &flag, sizeof(int) * (N + BLOCK - 1) / BLOCK);
    cudaMalloc((void**) &inter_values, sizeof(float) * (N + BLOCK - 1) / BLOCK);
    cudaMalloc((void**) &block_counter, sizeof(int));
    cudaMalloc((void**) &A_d, sizeof(float) * N);
    cudaMalloc((void**) &B_d, sizeof(float) * N);

    // Zeroing out flags and block_counter
    cudaMemset((void*) flag, 0, sizeof(int) * (N + BLOCK - 1) / BLOCK);
    cudaMemset((void*) block_counter, 0, sizeof(int));

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) { A_h[i] = generator(); }

    for(int i = 0; i < N; ++i) {
        if(i % BLOCK == 0) { printf("| "); }
        printf("%5.0f ", A_h[i]);
    }
    printf("\n");

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size( (N + BLOCK - 1) / BLOCK );
    dim3 block_size(BLOCK);

    scan_kogge_stone_domino_style<<<grid_size, block_size>>>(A_d, B_d, N, block_counter, flag, inter_values);

    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i) {
        if(i % BLOCK == 0) { printf("| "); }
        printf("%5.0f ", B_h[i]);
    }
    printf("\n\n");

    delete [] B_h;
    delete [] A_h;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(inter_values);
    cudaFree(block_counter);
    cudaFree(flag);

    return 0;
}



__global__ void scan_kogge_stone_domino_style(float *A, float *B, int N, int *block_counter, int *flag, float *inter_values) {
    
    __shared__ int block_id;
    __shared__ float passed_value;
    __shared__ float AB_sh[BLOCK];
    __shared__ int chunk_size;

    // Dynamic indexing of blocks
    if(threadIdx.x == 0) {
        block_id = atomicAdd(block_counter, 1);
        chunk_size = (block_id+1) * BLOCK > N ? N - block_id * BLOCK : BLOCK;
    }
    __syncthreads();

    // Coping global data to shared memory
    if(threadIdx.x < chunk_size) {
        AB_sh[threadIdx.x] = A[block_id * BLOCK + threadIdx.x];
    } else {
        AB_sh[threadIdx.x] = 0.0f;
    }

    // Standard Kogge-Stone algorithm for block
    float temp_value;
    for(int stride = 1; stride < chunk_size; stride *= 2) {
        if(threadIdx.x >= stride && threadIdx.x < chunk_size) {
            temp_value = AB_sh[threadIdx.x] + AB_sh[threadIdx.x - stride];
        }
        __syncthreads();

        AB_sh[threadIdx.x] = temp_value;
    }

    // Loading passed value from the global array
    if(threadIdx.x == 0 && block_id != 0) {    
        while(atomicAdd(&flag[block_id-1], 0) == 0) { /*waiting for value to change*/ }
        passed_value = inter_values[block_id-1];
    } else if(threadIdx.x == 0 && block_id == 0) {
        passed_value = 0.0f;
    }
    __syncthreads();

    // Saving local scan value with added passed value
    if(threadIdx.x == 0) {
        inter_values[block_id] = AB_sh[chunk_size-1] + passed_value;
        __threadfence(); // To make sure that flag value change is visible after passing scan result
        atomicAdd(&flag[block_id], 1);
    }

    // Adding passed_value to all local scan results
    if(threadIdx.x < chunk_size) {
        AB_sh[threadIdx.x] += passed_value;
        B[block_id * BLOCK + threadIdx.x] = AB_sh[threadIdx.x];
    }
}