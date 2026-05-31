#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 32

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
    dim3 grid_size((num_threads + BLOCK - 1) / BLOCK); // correct ceiling operator in other codes
    
    reduction_basic<<<grid_size, block_size>>>(A_d, B_d, N);

    cudaMemcpy(&B_h, B_d, sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i) {
        printf("%3.0f ", A_h[i]);
    }
    printf("\nReduction result: %f", B_h);

    cudaFree(A_d);
    cudaFree(B_d);
    delete [] A_h;

    return 0;
}


__global__ void reduction_basic(float *A, float *B, int N){
    // each block consumes 2*blockDim elements so it needs to be multiplied accordingly
    int block_start = 2 * blockIdx.x * blockDim.x;
    int id = block_start + threadIdx.x;
    
    // in case of odd length of stride sometimes it needs to add one neutral element at the end
    int prev_stride = 4 * blockDim.x;
    int stride = 2 * blockDim.x;
    do{    
        prev_stride = stride;
        stride = (stride + 1) / 2;

        if(id < block_start + stride && id + stride < N) {
            if(threadIdx.x + stride < prev_stride) {
                A[id] += A[id + stride];
            }
        }
        __syncthreads();
    } while (stride > 1);



    if(threadIdx.x == 0) {
        atomicAdd(B, A[id]);
    }
}
