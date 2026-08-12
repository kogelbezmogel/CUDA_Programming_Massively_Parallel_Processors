#include <iostream>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "multiply_std.h"

#define TILE_SIZE 16


__global__ void kernel_simple(float *A, float *B, float *C, int N, int K, int M) {
    
    // memory is of this size because tail can be loaded and used to calculations after which those values
    // won't be usefull in context of block. Therefore, tail can be overwritten. Each load and calculation is named phase
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float value = 0;

    for(int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {

        if(tx + phase*TILE_SIZE < K && row < N) {
            A_tile[ty][tx] = A[row*K + tx + phase*TILE_SIZE];
        }
        else {
            A_tile[ty][tx] = 0;
        }

        if(col < M && ty + phase*TILE_SIZE < K) {
            B_tile[ty][tx] = B[(ty + phase*TILE_SIZE) * M + col];
        } 
        else {
            B_tile[ty][tx] = 0;
        }
        __syncthreads(); // before next step all threads must load full tail from phase

        for(int i = 0; i < TILE_SIZE; ++i) {
            value += A_tile[ty][i] * B_tile[i][tx];
        }
        __syncthreads();
    }

    C[row * M + col] = value;
}


__global__ void kernel_dynamic(float *A, float *B, float *C, int N, int K, int M, size_t tile_size) {
    // extern array will be definite by the third kernel parameter "tile_memory_usage"
    extern __shared__ char A_and_B_tile[];
    float *A_tile = (float*) A_and_B_tile;
    float *B_tile = (float*) A_and_B_tile + tile_size * tile_size;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float value = 0;
    for(int phase = 0; phase < (K + tile_size - 1) / tile_size; ++phase) {

        if(tx + phase*tile_size < K && row < N) {
            A_tile[ty * tile_size + tx] = A[row*K + tx + phase*tile_size];
        }
        else {
            A_tile[ty * tile_size + tx] = 0;
        }

        if(col < M && ty + phase*tile_size < K) {
            B_tile[ty * tile_size + tx] = B[(ty + phase*tile_size) * M + col];
        } 
        else {
            B_tile[ty * tile_size + tx] = 0;
        }
        __syncthreads();

        for(int i = 0; i < tile_size; ++i) {
            value += A_tile[ty * tile_size + i] * B_tile[i * tile_size + tx];
        }

        if(row < N && col < M)
            C[row * M + col] = value;
        __syncthreads();
    }
}


int calculate_appropriate_SM_usage() {
    // return amount of bytes that can be used per block for mximum efficency
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    size_t max_size = floor( sqrt(devProp.maxThreadsPerBlock) );
    
    // checking how many threads can be used in block regarding shared memory
    // algorithm uses 8T^2 bytes per block of size TxT
    size_t size = floor( sqrt(devProp.sharedMemPerBlock/8) );
    if(size < max_size)
        max_size = size;

    return max_size;
}


int main() {
    float *A_h, *B_h, *C_h, *C_e;
    float *A_d, *B_d, *C_d;

    int N, K, M;
    N = 4; 
    K = 4;
    M = 4;

    A_h = new float[N * K];
    B_h = new float[K * M];
    C_h = new float[N * M];
    C_e = new float[N * M];

    for(int i = 0; i < N * K; ++i)
        A_h[i] = (i*i*i + 5*i) % 41;

    for(int i = 0; i < K * M; ++i)
        B_h[i] = (i*i*i + 10*i) % 41;
    multiply_matrices_cpu(A_h, B_h, C_e, N, K, M);

    cudaMalloc((void**) &A_d, N*K*sizeof(float));
    cudaMalloc((void**) &B_d, K*M*sizeof(float));
    cudaMalloc((void**) &C_d, N*M*sizeof(float));
    cudaMemcpy(A_d, A_h, N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K*M*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_size( ceil((float) M / TILE_SIZE), ceil((float) N / TILE_SIZE), 1);
    kernel_simple<<<grid_size, block_size>>>(A_d, B_d, C_d, N, K, M);
    cudaMemcpy(C_h, C_d, N*M*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N * M; ++i) {
        assert( C_e[i] - C_h[i]  < 1e-4 );
    }

    size_t tile_size = calculate_appropriate_SM_usage();
    size_t tiles_memory = 2 * sizeof(float) * tile_size * tile_size;
    dim3 block_size_dyn(tile_size, tile_size, 1);
    dim3 grid_size_dyn(ceil((float) M / tile_size), ceil((float) N / tile_size), 1);
    kernel_dynamic<<<grid_size_dyn, block_size_dyn, tiles_memory>>>(A_d, B_d, C_d, N, K, M, tile_size);
    cudaMemcpy(C_h, C_d, N*M*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N * M; ++i) {
        assert( C_e[i] - C_h[i]  < 1e-4 );
    }


    delete [] C_e;
    delete [] C_h;
    delete [] A_h;
    delete [] B_h;

    cudaFree(C_d);
    cudaFree(A_d);
    cudaFree(B_d);
    std::cout << "PASSED\n";
    return 0;
}