#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 4

__global__ void scan_kogge_stone_v2(float *A, float *B, int N);



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
        printf("%5.1f ", A_h[i]);
    }
    printf("\n");

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size( (N + BLOCK - 1) / BLOCK );
    dim3 block_size(BLOCK);
    scan_kogge_stone_v2<<<grid_size, block_size>>>(A_d, B_d, N);
    
    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    delete [] B_h;
    delete [] A_h;
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}


__global__ void scan_kogge_stone_v2(float *A, float *B, int N) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // add shared mamemory single buffer AB as in basic version

    if(id < N) {
        B[id] = A[id];
    }

    float temp;
    for(int stride = 1; stride < BLOCK; stride *= 2) {

        __syncthreads();
        if(threadIdx.x >= stride && id < N) { 
            temp = B[id] + B[id - stride];
        }

        __syncthreads();
        if(threadIdx.x >= stride && id < N) {
            B[id] = temp;
        }
    }

    __syncthreads();
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        for(int i = 0; i < BLOCK; ++i) {
            printf("%5.1f ", B[blockIdx.x * blockDim.x + i]);
        }
        printf("\n");
    }
}