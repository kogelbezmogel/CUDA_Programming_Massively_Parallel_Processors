#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 2
#define CFACTOR 3

__global__ void reduction_coarsed(float *A, float *B, int N);


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
    int num_threads = (N + CFACTOR-1) / (CFACTOR);
    dim3 grid_size((num_threads + BLOCK - 1) / BLOCK); // correct ceiling operator in other codes

    
    reduction_coarsed<<<grid_size, block_size>>>(A_d, B_d, N);

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


__global__ void reduction_coarsed(float *A, float *B, int N){
    
    // For Cfactor==2 algorithm works almost exacly like the basic one but without shared memory
    int block_start = CFACTOR * blockIdx.x * blockDim.x;
    int id = block_start + threadIdx.x;
    
    __shared__ float A_sh[BLOCK];

    // initializing data with first row
    if(id < N) {
        A_sh[threadIdx.x] = A[id];
    } else {
        A_sh[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // Sequentialy reducting CFACTOR*BLOCK size piece to BLOCK size piece
    for(int step = 1; step < CFACTOR; ++step) {
        if(id + step * BLOCK < N) {
            A_sh[threadIdx.x] += A[id + step * BLOCK];
        }
        __syncthreads();
    }

    // Basic algorithm on BLOCK size chunk
    int curr_stride = BLOCK;
    int prev_stride = 2*BLOCK;
    do {
        prev_stride = curr_stride;
        curr_stride = (curr_stride+1) / 2;
        
        if(threadIdx.x < curr_stride) {
            if (threadIdx.x + curr_stride < prev_stride) {
                A_sh[threadIdx.x] += A_sh[threadIdx.x + curr_stride];
            }
        }
        __syncthreads();

    } while (curr_stride > 1);

    // accumulating blocks
    if(threadIdx.x == 0) {
        // printf("(Block %d val:%f)",blockIdx.x, A_sh[0]);
        atomicAdd(B, A_sh[threadIdx.x]);
    }
}
