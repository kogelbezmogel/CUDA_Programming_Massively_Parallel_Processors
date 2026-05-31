// a) Write a kernel that has each thread produce one output matrix column
// b) Write a kernel that has each thread produce one output matrix row

#include <iostream>
#include <math.h>
#include <assert.h>
#include "multiply_std.h"


__global__ void multiply_matrices_cell_per_thread(float *A, float *B, float *C, int N, int K, int M) {
    float s_value;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if( row < N && col < M) {
        s_value = 0;
        for(int k = 0; k < K; ++k) {
            s_value += A[row*K + k] * B[k*M + col];
        }
        C[row*M + col] = s_value;
    }
}


// a)
__global__ void multiply_matrices_column_per_thread(float *A, float *B, float *C, int N, int K, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s_value;
    if(col < M) {
     
        for(int row = 0; row < N; ++row) {
     
            s_value = 0;
            for(int k = 0; k < K; ++k) {
                s_value += A[row*K + k] * B[k*M + col];
            }
            C[row*M + col] = s_value;
        }
    }
}


// b)
__global__ void multiply_matrices_row_per_thread(float *A, float *B, float *C, int N, int K, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float s_value;

    if(row < N) {
        for(int col = 0; col < M; ++col) {
            s_value = 0;
            for(int k = 0; k < K; ++k) {
                s_value += A[row*K + k] * B[k*M + col];
            }
            C[row*M + col] = s_value;
        }
    }

}


int main() {

    int N = 20;
    int K = 30;
    int M = 40;

    float *A_h, *B_h, *C_h, *C_e;
    float *A_d, *B_d, *C_d;

    A_h = new float[N * K];
    B_h = new float[K * M];
    C_h = new float[N * M];
    C_e = new float[N * M];

    cudaMalloc((void**) &A_d, N * K * sizeof(float));
    cudaMalloc((void**) &B_d, K * M * sizeof(float));
    cudaMalloc((void**) &C_d, N * M * sizeof(float));

    // multiplying matrices on cpu
    multiply_matrices_cpu(A_h, B_h, C_e, N, K, M);

    cudaMemcpy(A_d, A_h, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K * M * sizeof(float), cudaMemcpyHostToDevice);

    // standard method
    dim3 block_grid(ceil((float) M / 32), ceil((float) N / 32), 1);
    dim3 block_dim(32, 32, 1);
    multiply_matrices_column_per_thread<<<block_grid, block_dim>>>(A_d, B_d, C_d, N, K, M);
    cudaMemcpy(C_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    for(int m = 0; m < N * M; ++m)
        assert( C_e[m] - C_h[m] < 1e-4 );

    // each thread computating one column
    block_grid = dim3(ceil((float) M / 256), 1, 1);
    block_dim = dim3(256, 1, 1);
    multiply_matrices_column_per_thread<<<block_grid, block_dim>>>(A_d, B_d, C_d, N, K, M);
    cudaMemcpy(C_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    for(int m = 0; m < N * M; ++m)
        assert( C_e[m] - C_h[m] < 1e-4 );

    //each thread computating one row
    block_grid = dim3(1, ceil((float) N / 256), 1);
    block_dim = dim3(1, 256, 1);
    multiply_matrices_row_per_thread<<<block_grid, block_dim>>>(A_d, B_d, C_d, N, K, M);
    cudaMemcpy(C_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    for(int m = 0; m < N * M; ++m)
        assert( C_e[m] - C_h[m] < 1e-4 );

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    delete [] A_h;
    delete [] B_h;
    delete [] C_h;
    delete [] C_e;

    std::cout << "PASSED\n";
    
    return 0;
}