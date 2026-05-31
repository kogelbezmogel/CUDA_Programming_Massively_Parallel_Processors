#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 32


__global__ void scan_kogge_stone(float *A, float *B, int N);


int main() {
    float *A_h, *A_d, *B_h, *B_d;
    int N = 12;
    A_h = new float[N];
    B_h = new float[N];

    cudaMalloc((void**) &A_d, sizeof(float) * N);
    cudaMalloc((void**) &B_d, sizeof(float) * N);

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) { A_h[i] = generator(); }

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size( (N + BLOCK - 1) / BLOCK );
    dim3 block_size(BLOCK);
    scan_kogge_stone<<<grid_size, block_size>>>(A_d, B_d, N);
    
    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    delete [] B_h;
    delete [] A_h;
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}


__global__ void scan_kogge_stone(float *A, float *B, int N) {

    __shared__ float AB[BLOCK];


    if(threadIdx.x < N) {
        AB[threadIdx.x] = A[threadIdx.x];
    } else {
        AB[threadIdx.x] = 0;
    }

    float temp;
    for(int stride = 1; stride < N; stride *= 2) {

        __syncthreads();
        if(threadIdx.x >= stride && threadIdx.x < N) { 
            temp = AB[threadIdx.x] + AB[threadIdx.x - stride];
        }
        __syncthreads();
        AB[threadIdx.x] = temp;
    }


    if(threadIdx.x < N) {
        B[threadIdx.x] = AB[threadIdx.x];
    }
}

