#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 32

__global__ void scan_brent_kung(float *A, float *B, int N);


int main() {
    float *A_h, *A_d, *B_h, *B_d;
    int N = 8;
    A_h = new float[N];
    B_h = new float[N];

    cudaMalloc((void**) &A_d, sizeof(float) * N);
    cudaMalloc((void**) &B_d, sizeof(float) * N);

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) { A_h[i] = generator(); }

    for(int i = 0; i < N; ++i) {
        if(i % BLOCK == 0) { printf("| "); }
        printf("%5.0f ", A_h[i]);
    }
    printf("\n\n");

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size( (N + BLOCK - 1) / BLOCK );
    dim3 block_size(BLOCK);
    scan_brent_kung<<<grid_size, block_size>>>(A_d, B_d, N);
    
    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i) {
        if(i % BLOCK == 0) { printf("| "); }
        printf("%5.0f ", B_h[i]);
    }
    printf("\n\n");

    delete [] A_h;
    delete [] B_h;
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}


__global__ void scan_brent_kung(float *A, float *B, int N) {
    __shared__ float AB_sh[BLOCK];
    int id_mapped;

    if(threadIdx.x < N) {
        AB_sh[threadIdx.x] = A[threadIdx.x];
    } else {
        AB_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Reduction phase
    int max_stride;
    for(int stride = 1; stride < N; stride *= 2) {
        id_mapped = (threadIdx.x+1) * stride*2 - 1;
        if( (id_mapped+1) % (2*stride) == 0  && id_mapped < N) {
            AB_sh[id_mapped] += AB_sh[id_mapped - stride];
        }
        max_stride = stride;
    }

    // Inverse tree phase
    for(int stride = max_stride/2; stride > 0; stride /= 2) {
        id_mapped = (threadIdx.x+1) * stride*2 - 1;
        if( (id_mapped+1) % (2*stride) == 0 && id_mapped + stride < N) {
            AB_sh[id_mapped + stride] += AB_sh[id_mapped];
        }
    }
    
    if(threadIdx.x < N) {
        B[threadIdx.x] = AB_sh[threadIdx.x];
    }
}