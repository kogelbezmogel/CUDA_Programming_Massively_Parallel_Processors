#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 5
#define CFACTOR 4

__global__ void scan_brent_kung(float *A, float *B, int N);


int main() {
    float *A_h, *A_d, *B_h, *B_d;
    int N = 20;
    A_h = new float[N];
    B_h = new float[N];

    cudaMalloc((void**) &A_d, sizeof(float) * N);
    cudaMalloc((void**) &B_d, sizeof(float) * N);

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) { A_h[i] = generator(); }

    for(int i = 0; i < N; ++i) {
        printf("%5.0f ", A_h[i]);
    }
    printf("\n\n");

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size(1);
    dim3 block_size(BLOCK);
    scan_brent_kung<<<grid_size, block_size>>>(A_d, B_d, N);
    
    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i) {
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
    __shared__ float AB_sh[BLOCK * CFACTOR];
    int id_mapped;

    // Loading data to shared memory
    for(int i = 0; i < CFACTOR; ++i) {
        if(threadIdx.x + i*BLOCK < N) {
            AB_sh[threadIdx.x + i*BLOCK] = A[threadIdx.x + i*BLOCK]; 
        } else {
            AB_sh[threadIdx.x + i*BLOCK] = 0.0f;
        }
    }
    __syncthreads();

    // Sequential computation on CFACTOR elements
    for(int i = 1; i < CFACTOR; ++i) {
        if(threadIdx.x * CFACTOR + i < N) {
            AB_sh[threadIdx.x * CFACTOR + i] += AB_sh[threadIdx.x * CFACTOR + i-1];
        }
    }

    // Reduction phase
    int max_stride;
    for(int stride = 1; stride * CFACTOR < N; stride *= 2) {
        id_mapped = (threadIdx.x+1) * CFACTOR * stride*2 - 1;
        if(id_mapped < N) {
            AB_sh[id_mapped] += AB_sh[id_mapped - stride * CFACTOR];
        }
        max_stride = stride;
    }
    __syncthreads();

    // Inverse tree phase
    for(int stride = max_stride/2; stride > 1; stride /= 2) {
        id_mapped = (threadIdx.x+1) * CFACTOR * stride - 1;
        if(id_mapped + stride / 2 * CFACTOR  < N) {
            AB_sh[id_mapped + stride / 2 * CFACTOR] += AB_sh[id_mapped];
        }
    }
    __syncthreads();
    // After the inverse tree phase each last element in segment has accumulated sum from all previous elements.
    // It is enaugh to add those accumullated values to the next segments.

    // Adding last value from segment to the CFACTOR-1 elements from next segment 
    id_mapped = (threadIdx.x + 1) * CFACTOR - 1;
    for(int i = 1; i < CFACTOR; ++i) {
        if(id_mapped + i < N) {
            AB_sh[id_mapped + i] += AB_sh[id_mapped];       
        }
    }
    __syncthreads();

    // Transfering data from shared memory to the B array
    for(int i = 0; i < CFACTOR; ++i) {
        if(threadIdx.x + i*BLOCK < N) {
            B[threadIdx.x + i*BLOCK] = AB_sh[threadIdx.x + i*BLOCK];
        }
    }
}