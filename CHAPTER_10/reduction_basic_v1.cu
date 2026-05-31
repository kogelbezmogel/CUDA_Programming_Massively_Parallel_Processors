#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 4

__global__ void reduction_basic(float *A, float *B, int N);


int main() {

    float *A_h, *A_d;
    float *B_d;
    float B_h;
    int N = 10;

    A_h = new float[N];
    cudaMalloc((void**) &A_d, sizeof(float) * N);
    cudaMalloc((void**) &B_d, sizeof(float));

    RandomIntGenerator generator(967, 0, 10);
    for(int i = 0; i < N; ++i) {
        A_h[i] = generator();
    }

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    
    dim3 block_size(BLOCK);
    int num_threads = (N + 1) / 2;
    dim3 grid_size((num_threads + BLOCK) / BLOCK);
    
    reduction_basic<<<grid_size, block_size>>>(A_d, B_d, N);

    cudaMemcpy(&B_h, B_d, sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i) {
        printf("%f ", A_h[i]);
    }
    printf("\nReduction result: %f", B_h);

    cudaFree(A_d);
    cudaFree(B_d);
    delete [] A_h;

    return 0;
}


__global__ void reduction_basic(float *A, float *B, int N){
    
    int id = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    for(int stride = 1; stride < 2 * blockDim.x; stride *= 2) { 
        if(threadIdx.x % stride == 0 && id + stride < N) {
            A[id] += A[id + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        atomicAdd(B, A[id]);
    }
}
