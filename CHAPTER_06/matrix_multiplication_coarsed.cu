#include "matrix_multiplication_coarsed.cuh"

#include <stdio.h>

#define CFACTOR 4
#define TILE 16


/*
Kernel statistics:
    8*TILE*TILE B shared memory per block
    21 Registers per thread
*/





void __global__ matrix_multiplication_coarsed(float *A, float *B, float *C, int N, int K, int M) {
    /*
        The choice of this implementaion of
        coarsing is to serialise loading matrix B
    */
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    float temp_values[CFACTOR];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i = 0; i < CFACTOR; ++i) {
        temp_values[i] = 0.0f;
    }

    for(int phase = 0; phase < (K + TILE - 1) / TILE; ++phase) {

        // loading A_tile
        if(row < N && threadIdx.x + phase * blockDim.x < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + threadIdx.x + phase * blockDim.x];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }


        for(int coarse = 0; coarse < CFACTOR; ++coarse) {

            // loading B_tile 
            if(threadIdx.y + phase * blockDim.y < K && col + coarse * blockDim.x < M) {
                B_tile[threadIdx.y][threadIdx.x] = B[(threadIdx.y + phase * blockDim.y) * M + col + coarse * blockDim.x];
            } else {
                B_tile[threadIdx.y][threadIdx.x] = 0.0f;
            } 
            __syncthreads();

            for(int k = 0; k < K; ++k) {
                temp_values[coarse] += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
            }
            __syncthreads();
        }
    }

    for(int coarse = 0; coarse < CFACTOR; ++coarse) {
        if(row < N && col + coarse * TILE < M) {
            C[row * M + col + coarse * TILE] = temp_values[coarse];
        }
    }

}