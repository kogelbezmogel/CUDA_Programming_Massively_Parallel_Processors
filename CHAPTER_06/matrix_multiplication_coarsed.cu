#include "matrix_multiplication_coarsed.cuh"

#include <stdio.h>

#define MAX_CFACTOR 16
#define TILE 4


/*
Kernel statistics:
    8*TILE*TILE B shared memory per block
    23 Registers per thread
*/





void __global__ matrix_multiplication_coarsed(float *A, float *B, float *C, int N, int K, int M, int cf) {
    /*
        The choice of this implementaion of
        coarsing is to serialise loading matrix B
    */
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    float temp_values[MAX_CFACTOR];

    int row = blockDim.y * blockIdx.y + threadIdx.y;

    for(int i = 0; i < cf; ++i) {
        temp_values[i] = 0.0f;
    }

    for(int phase = 0; phase < (K + TILE - 1) / TILE; ++phase) {

        // loading A_tile
        if(row < N && threadIdx.x + phase * blockDim.x < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + threadIdx.x + phase * blockDim.x];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        for(int coarse = 0; coarse < cf; ++coarse) {

            // loading B_tile 
            /*
                Each block computes
                    blockDim.x * cf     values (horizontally)
                    blockDim.y          values (vertically)

                Each thread in block computes cf values. Each value is in the
                same row but separeted BlockDim.x columns from previous one

                That is why value are being loaded from range
                [blockId.x * blockDim.x * cf;   (blockId.x+1) * blockDim.x * cf - 1]
            */
            if(threadIdx.y + phase * blockDim.y < K && (blockIdx.x * cf + coarse) * blockDim.x + threadIdx.x < M) {
                B_tile[threadIdx.y][threadIdx.x] = B[(threadIdx.y + phase * blockDim.y) * M + (blockIdx.x * cf + coarse) * blockDim.x + threadIdx.x];
            } else {
                B_tile[threadIdx.y][threadIdx.x] = 0.0f;
            } 
            __syncthreads();

            for(int k = 0; k < TILE; ++k) {
                temp_values[coarse] += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
            }
            __syncthreads();
        }
    }

    for(int coarse = 0; coarse < cf; ++coarse) {
        if(row < N && (blockIdx.x * cf + coarse) * blockDim.x + threadIdx.x < M) {
            C[row * M + (blockIdx.x * cf + coarse) * blockDim.x + threadIdx.x] = temp_values[coarse];
        }
    }

}