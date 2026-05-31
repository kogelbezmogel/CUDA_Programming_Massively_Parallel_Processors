#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 32


__global__ void scan_kogge_stone_double_buffer(float *A, float *B, int N);


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
    scan_kogge_stone_double_buffer<<<grid_size, block_size>>>(A_d, B_d, N);
    
    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    delete [] B_h;
    delete [] A_h;
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}


__global__ void scan_kogge_stone_double_buffer(float *A, float *B, int N) {

    __shared__ float AB_1[BLOCK];
    __shared__ float AB_2[BLOCK];

    float *AB_1p = AB_1;
    float *AB_2p = AB_2;
    float *temp;

    if(threadIdx.x == 0) {
        for(int i = 0; i < N; ++i) {
            printf("%2.0f ", A[i]);
        }
        printf("\n");
    }

    if(threadIdx.x < N) {
        AB_1p[threadIdx.x] = A[threadIdx.x];
    } else {
        AB_1p[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int stride = 1; stride < N; stride *= 2) {

        if(threadIdx.x >= stride && threadIdx.x < N) { 
            AB_2p[threadIdx.x] = AB_1p[threadIdx.x] + AB_1p[threadIdx.x - stride];
        } else {
            AB_2p[threadIdx.x] = AB_1p[threadIdx.x]; //instead of coping all unupdated values maybe it would be efficient to copy only freshly finished values. If value is once finished it won't change. It need to be copied only once to second buffer to be preserved.
        }
    
        // swaping pointers to read from last used
        temp = AB_1p;
        AB_1p = AB_2p;
        AB_2p = temp;

        __syncthreads();
    }

    if(threadIdx.x < N) {
        B[threadIdx.x] = AB_1p[threadIdx.x];
    }

    if(threadIdx.x == 0) {
        for(int i = 0; i < N; ++i) {
            printf("%2.0f ", B[i]);
        }
        printf("\n");
    }
}

