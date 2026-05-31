#include <iostream>
#include <stdio.h>
#include "RandomIntGenerator.h"

#define BLOCK 8
#define BINS 10
#define CFACTOR 4

__global__ void histogram_coarsed(int *A, int *H, int N);

int main() {

    int *A_h, *A_d, *H_h, *H_d;
    int N = 100;

    A_h = new int[N];
    H_h = new int[10];

    cudaMalloc((void**) &A_d, sizeof(int) * N);
    cudaMalloc((void**) &H_d, sizeof(int) * 10);

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) {
        A_h[i] = generator();
    }

    cudaMemcpy(A_d, A_h, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(H_d, H_h, sizeof(int) * 10, cudaMemcpyHostToDevice);

    dim3 grid_size( (N + CFACTOR * BLOCK) / (CFACTOR * BLOCK) );
    dim3 block_size(BLOCK);
    histogram_coarsed<<<grid_size, block_size>>>(A_d, H_d, N);

    cudaMemcpy(H_h, H_d, sizeof(int) * 10, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i) {
        printf("%d ", A_h[i]);
    }
    printf("\n");
    for(int i = 0; i < 10; ++i) {
        printf("%d ", H_h[i]);
    }

    delete [] A_h;
    delete [] H_h;
    cudaFree(A_d);
    cudaFree(H_d);

    return 0;
}


__global__ void histogram_coarsed(int *A, int *H, int N) {
    int id_start = blockIdx.x * blockDim.x * CFACTOR + threadIdx.x;

    __shared__ int hist_sh[BINS];

    // setting shared bins at 0
    for(int bin = threadIdx.x; bin <  BINS; bin += blockDim.x) {
        hist_sh[bin] = 0;
    }
    __syncthreads();

    // mapping CFACTOR elements to bins. Loading pattern takes advantage of memory burst
    for(int id = id_start; id < (blockIdx.x+1) * blockDim.x * CFACTOR; id += blockDim.x) {
        if(id < N) {
            int bin_id = A[id];
            atomicAdd(hist_sh + bin_id, 1);
        }
    }
    __syncthreads();

    // aggregating results from blocks to global memory
    for(int bin = threadIdx.x; bin <  BINS; bin += blockDim.x) {
        atomicAdd(H + bin, hist_sh[bin]);
    }
}
