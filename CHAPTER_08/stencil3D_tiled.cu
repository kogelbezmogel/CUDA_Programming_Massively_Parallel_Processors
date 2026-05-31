#include <iostream>
#include <stdio.h>
#include <math.h>
#include "../UTILS/utils.h"
#include "../UTILS/stencil_sequential.h"

void __global__ stencil3D_tiled(float *in, float *out, float *W, int N, int K, int M, int rank);
void __host__ __device__ print_tensor(float *A, int N, int K, int M);

#define T 8

int main() {

    int N, K, M, rank, Dim;
    rank = 1;
    N = 5;
    K = 5;
    M = 5;
    Dim = 3; 

    float *A_h, *B_h, *W_h, *A_d, *B_d, *W_d;
    A_h = new float[N * K * M];
    B_h = new float[N * K * M];
    W_h = new float[Dim * 2*rank + 1];
    cudaMalloc((void**) &A_d, sizeof(float) * N*K*M);
    cudaMalloc((void**) &B_d, sizeof(float) * N*K*M);
    cudaMalloc((void**) &W_d, sizeof(float) * (Dim * 2 * rank + 1));

    dim3 block_size(T, T, T);
    dim3 grid_size((K+T-1)/T, (N+T-1)/T, (M+T-1)/T);
    
    fill_tensor_with_data(A_h, N, K, M);
    fill_stencil_weights(W_h, rank);
    cudaMemcpy(A_d, A_h, sizeof(float)*N*K*M, cudaMemcpyHostToDevice);
    cudaMemcpy(W_d, W_h, sizeof(float)*(Dim*2*rank+1), cudaMemcpyHostToDevice);

    stencil3D_tiled<<grid_size, block_size>>(A_d, B_d, W_d, N, K, M, rank);

    cudaMemcpy(B_h, B_d, sizeof(float)*N*K*M, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(W_d);
    delete [] A_h;
    delete [] B_h;
    delete [] W_h;

    return 0;
}


void __global__ stencil3D_tiled(float *in, float *out,float *W, int N, int K, int M, int rank){
    // Tile IN  T
    // Tile OUT T-2*rank
    int row = blockIdx.y*(T-2*rank) + threadIdx.y;
    int col = blockIdx.x*(T-2*rank) + threadIdx.x;
    int pla = blockIdx.z*(T-2*rank) + threadIdx.z;

    __shared__ float *in_sh[T][T][T];
    
    //loading data to shared memory
    if(row < N && col < K && pla < M) {
        in_sh[threadIdx.z][threadIdx.y][threadIdx.x] = out[pla * N*K + row * K + col];
    }
    __syncthreads();

    // computing data
    // 1. Check if thread in block is destined to compute. Excluding rank threads at each border of block
    // 2. Check if thread that is destined to compute is still inside the input data. Prevents threads from reaching outside the data.
    if(threadIdx.x - rank >= 0 && threadIdx.x + rank < T && threadIdx.y - rank >= 0 && threadIdx.y + rank < T && threadIdx.z - rank >= 0 && threadIdx.z + rank < T) {
        if(row + rank < N && col + rank < K && pla + rank < M) {

            out[pla * N*K + row * K + col] = W[0] * in_sh[threadIdx.z][threadIdx.y][threadIdx.x]
                                           + W[1] * in_sh[threadIdx.z][threadIdx.y][threadIdx.x+1]
                                           + W[2] * in_sh[threadIdx.z][threadIdx.y][threadIdx.x-1]
                                           + W[3] * in_sh[threadIdx.z][threadIdx.y+1][threadIdx.x]
                                           + W[4] * in_sh[threadIdx.z][threadIdx.y-1][threadIdx.x]
                                           + W[5] * in_sh[threadIdx.z+1][threadIdx.y][threadIdx.x]
                                           + W[6] * in_sh[threadIdx.z-1][threadIdx.y][threadIdx.x];
        }
    }
}


void __host__ __device__ print_tensor(float *A, int N, int K, int M) {
    for(int m = 0; m < M; ++m) {
        printf("[\n");
        for(int n = 0; n < N; ++n) {
            printf("    [");
            for(int k = 0; k < K-1; ++k) {
                printf("%4.1f ", A[m*K*N + n*K + k]);
            }
            printf("%4.1f ]\n", A[m*K*N + n*K + K-1]);
        }
        printf("]\n");
    }
}


// Is better to used shared memory for W[] or depend on cashed memory?