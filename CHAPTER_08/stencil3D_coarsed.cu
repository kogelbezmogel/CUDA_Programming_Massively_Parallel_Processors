#include <iostream>
#include <stdio.h>
#include <math.h>
#include "../UTILS/utils.h"
#include "../UTILS/stencil_sequential.h"

void __global__ stencil3D_coarsed(float *in, float *out, float *W, int N, int K, int M, int rank);
void __host__ __device__ print_tensor(float *A, int N, int K, int M);

#define T 32

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


void __global__ stencil3D_coarsed(float *in, float *out,float *W, int N, int K, int M, int rank){
    /* Here the algorithm asserts that rank is equal 1.
    For bigger ranks there need to be some mechanism balancing the amount of used shared memory */
    if(rank != 1) {
        printf("ERROR: rank cannot be != 1");
        break;
    }
    

    // Tile IN  T
    // Tile OUT T-2*rank
    int row = blockIdx.y * (T-2*rank) + threadIdx.y;
    int col = blockIdx.x * (T-2*rank) + threadIdx.x;
    int pla_ite = blockIdx.z * (T-2*rank) + threadIdx.z;

    __shared__ float *in_prev[T][T];
    __shared__ float *in_curr[T][T];
    __shared__ float *in_next[T][T];
    

    //loading first plane at the begging
    if(pla_ite - 1 >= 0 && pla_ite - 1 < M) {
        if(row >= 0 && row < N && col >= 0 && col < K) {
            in_prev[row][col] = in[(pla_ite-1) * N*K + row * K + col];
        }
    }

    //loading second plane
    if(pla_ite>= 0 && pla_ite < M) {
        if(row >= 0 && row < N && col >= 0 && col < K) {
            in_curr[row][col] = in[pla_ite * N*K + row * K + col];
        }
    }
    
    for(pla_ite; pla_ite < M-1; ++pla_ite) {
        if(pla_ite + 1 >= 0 && pla_ite + 1 < M && row < N && col < K) {
            in_next[row][col] = in[(pla_ite+1) * N*K + row * K + col];
        }
        __syncthreads();

        if(row-1 >= 0 && row+1 < N && col-1 >= 0 && col+1 < K) {
            out[pla_ite * N*K + row * K + col] = W[0] * in_curr[row][col]
                                               + W[1] * in_curr[row+1][col]
                                               + W[2] * in_curr[row-1][col]
                                               + W[3] * in_curr[row][col+1]
                                               + W[4] * in_curr[row][col-1]
                                               + W[5] * in_next[row][col]
                                               + W[6] * in_prev[row][col];
        }
        __syncthreads();
        in_prev[row][col] = in_curr[row][col];
        in_curr[row][col] = in_next[row][col];
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