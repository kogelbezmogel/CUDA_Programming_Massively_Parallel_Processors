#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 4
#define CFACTOR 4

__global__ void scan_kogge_stone_coarsed(float *A, float *B, int N);



int main() {
    float *A_h, *A_d, *B_h, *B_d;
    int N = 16;
    A_h = new float[N];
    B_h = new float[N];

    cudaMalloc((void**) &A_d, sizeof(float) * N);
    cudaMalloc((void**) &B_d, sizeof(float) * N);

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) { A_h[i] = generator(); }

    for(int i = 0; i < N; ++i) {
        printf("%5.1f ", A_h[i]);
    }
    printf("\n\n");

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size(1);
    dim3 block_size(BLOCK);
    scan_kogge_stone_coarsed<<<grid_size, block_size>>>(A_d, B_d, N);
    
    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i) {
        printf("%5.1f ", B_h[i]);
    }
    printf("\n\n");

    delete [] B_h;
    delete [] A_h;
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}


__global__ void scan_kogge_stone_coarsed(float *A, float *B, int N) {

    // another thing to consider is when N is not multiply of CFACTOR. Under those circumstances the 3rd phase will be changed

    __shared__ float AB[CFACTOR * BLOCK];

    int block_start_id = blockIdx.x * blockDim.x * CFACTOR;

    // loading data to shared memory
    // !!loading should take advantage od memory coalescing so improve that section!!
    for(int i = 0; i < CFACTOR; ++i) {
        if(block_start_id + threadIdx.x * CFACTOR + i < N) {
            AB[threadIdx.x * CFACTOR + i] = A[block_start_id + threadIdx.x * CFACTOR + i];
        } else {
            AB[threadIdx.x * CFACTOR + i] = 0.0f;
        }
    }
    __syncthreads();

    // computing scan sequentialy by thread for CFACTOR elements 
    int sequence_start_id = block_start_id + threadIdx.x * CFACTOR;
    for(int i = 1; i < CFACTOR; ++i) {
        AB[sequence_start_id + i] += AB[sequence_start_id + i - 1];
    }
    __syncthreads();

    // if(blockIdx.x == 0 && threadIdx.x == 0) {
    //     for(int i = 0; i < BLOCK * CFACTOR; ++i) {
    //         printf("%5.1f ", AB[blockIdx.x * blockDim.x + i]);
    //     }
    //     printf("\n\n");
    // }

    // kogge-stone on sequentialy computed sections
    // maybe this indexing could be more readabale?
    float temp;
    for(int stride = 1; stride * CFACTOR < N; stride *= 2) {

        __syncthreads();
        if(threadIdx.x >= stride && threadIdx.x * CFACTOR < N) { 
            temp = AB[threadIdx.x*CFACTOR + CFACTOR-1] + AB[threadIdx.x*CFACTOR + CFACTOR-1 - stride * CFACTOR];
        }

        __syncthreads();
        if(threadIdx.x >= stride && threadIdx.x * CFACTOR < N) {
            AB[threadIdx.x*CFACTOR + CFACTOR-1] = temp;
        }
    }

    // if(blockIdx.x == 0 && threadIdx.x == 0) {
    //     for(int i = 0; i < BLOCK * CFACTOR; ++i) {
    //         printf("%5.1f ", AB[blockIdx.x * blockDim.x + i]);
    //     }
    //     printf("\n\n");
    // }

    // adding results of kogge-stone to sections
    __syncthreads();
    if(threadIdx.x > 0 && threadIdx.x * CFACTOR < N) {
        for(int i = 0; i < CFACTOR-1; ++i) {
            AB[threadIdx.x * CFACTOR + i] += AB[threadIdx.x*CFACTOR - 1];
        }
    }

    // if(blockIdx.x == 0 && threadIdx.x == 0) {
    //     for(int i = 0; i < BLOCK * CFACTOR; ++i) {
    //         printf("%5.1f ", AB[blockIdx.x * blockDim.x + i]);
    //     }
    //     printf("\n\n");
    // }

    // copping results from shared memory
    // !! saving data should take advantage of memory coalescing !!
    for(int i = 0; i < CFACTOR; ++i) {
        if(threadIdx.x * CFACTOR + i < N) {
            B[threadIdx.x * CFACTOR + i] = AB[threadIdx.x * CFACTOR + i];   
        }
    }

}

