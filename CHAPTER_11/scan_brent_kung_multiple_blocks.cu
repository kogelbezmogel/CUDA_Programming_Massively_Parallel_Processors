#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 4

__global__ void scan_brent_kung_mblocks(float *A, float *B, float *S, int N) {
    __shared__ float AB_sh[BLOCK];
    __shared__ int segment_size;
    int id_mapped;
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // evaluating last index in the block
    if(threadIdx.x == 0) {
        segment_size = (blockIdx.x + 1) * blockDim.x < N ? blockDim.x : N - blockIdx.x * blockDim.x;
    }

    // Loading data to shared memory
    if(id < N) {
        AB_sh[threadIdx.x] = A[id]; 
    } else {
        AB_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Reduction phase
    int max_stride;
    for(int stride = 1; stride < segment_size; stride *= 2) {
        id_mapped = (threadIdx.x+1) * stride*2 - 1;
        if(id_mapped < N) {
            AB_sh[id_mapped] += AB_sh[id_mapped - stride];
        }
        max_stride = stride;
    }
    __syncthreads();

    // Inverse tree phase
    for(int stride = max_stride/2; stride > 0; stride /= 2) {
        id_mapped = (threadIdx.x+1) * stride * 2 - 1;
        if(id_mapped + stride < segment_size) {
            AB_sh[id_mapped + stride] += AB_sh[id_mapped];
        }
    }
    __syncthreads();
    
    if(threadIdx.x == 0) {
        S[blockIdx.x] = AB_sh[segment_size-1];
    }

    // Transfering data from shared memory to the B array
    if(id < N) {
        B[id] = AB_sh[threadIdx.x];
    }
}

__global__ void scan_brent_kung_sblock(float *B, float *S, int N) {

    __shared__ float S_sh[1024];
    __shared__ int segment_size;

    if(threadIdx.x == 0) {
        segment_size = (N + BLOCK - 1) / BLOCK;
    }
    __syncthreads();

    if(threadIdx.x < segment_size) {
        S_sh[threadIdx.x] = S[threadIdx.x];
    } else {
        S_sh[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Reduction phase
    int max_stride;
    int id_mapped;
    for(int stride = 1; stride < segment_size; stride *= 2) {
        id_mapped = (threadIdx.x+1) * stride*2 - 1;
        if(id_mapped < segment_size) {
            S_sh[id_mapped] += S_sh[id_mapped - stride];
        }
        max_stride = stride;
    }

    // Reverse tree phase
    for(int stride = max_stride/2; stride > 0; stride /= 2) {
        id_mapped = (threadIdx.x + 1) * stride*2 - 1;
        if(id_mapped + stride < segment_size) {
            S_sh[id_mapped + stride] += S_sh[id_mapped];
        }
    }

    if(threadIdx.x < segment_size) {
        S[threadIdx.x] = S_sh[threadIdx.x];
    }
    __syncthreads();

    for(int i = 0; i < BLOCK; ++i) {
        if(threadIdx.x > 0 && threadIdx.x < segment_size) {
            B[threadIdx.x * BLOCK + i] += S_sh[threadIdx.x];
        }
    }

}


int main() {
    float *A_h, *A_d, *B_h, *B_d, *S_h, *S_d;
    int N = 16;
    A_h = new float[N];
    B_h = new float[N];
    S_h = new float[(N + BLOCK - 1) / BLOCK];

    cudaMalloc((void**) &A_d, sizeof(float) * N);
    cudaMalloc((void**) &B_d, sizeof(float) * N);
    cudaMalloc((void**) &S_d, sizeof(float) * (N + BLOCK - 1) / BLOCK);

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) { A_h[i] = generator(); }

    for(int i = 0; i < N; ++i) {
        printf("%5.0f ", A_h[i]);
    }
    printf("\n\n");

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size_m((N + BLOCK - 1) / BLOCK);
    dim3 block_size_m(BLOCK);
    scan_brent_kung_mblocks<<<grid_size_m, block_size_m>>>(A_d, B_d, S_d, N);
    
    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i) {
        printf("%5.0f ", B_h[i]);
    }
    printf("\n\n\n");

    cudaMemcpy(S_h, S_d, sizeof(float) * (N + BLOCK - 1) / BLOCK, cudaMemcpyDeviceToHost);
    for(int i = 0; i < (N + BLOCK - 1) / BLOCK; ++i) {
        printf("%5.0f ", S_h[i]);
    }
    printf("\n\n");

    dim3 grid_size_s(1);
    dim3 block_size_s(1024);
    scan_brent_kung_sblock<<<grid_size_s, block_size_s>>>(B_d, S_d, N);

    cudaMemcpy(S_h, S_d, sizeof(float) * (N + BLOCK - 1) / BLOCK, cudaMemcpyDeviceToHost);
    for(int i = 0; i < (N + BLOCK - 1) / BLOCK; ++i) {
        printf("%5.0f ", S_h[i]);
    }
    printf("\n\n");

    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i) {
        printf("%5.0f ", B_h[i]);
    }
    printf("\n\n\n");

    delete [] A_h;
    delete [] B_h;
    delete [] S_h;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(S_d);

    return 0;
}