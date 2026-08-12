#include "matrix_multiplication_tiled.cuh"

#include <iostream>
#include "stdio.h"

#define TILE 32

void __global__ matrix_multiplication_tiled(float *A, float *B, float *C, int N, int K, int M) {
    float __shared__ A_tile[TILE][TILE];
    float __shared__ B_tile[TILE][TILE];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float value = 0;

    for(int phase = 0; phase < (K + TILE - 1) / TILE; ++phase) {

        if(row < N && col + phase*TILE < K) {
            A_tile[row][col] = A[row * K + col + phase*TILE];
        } else {
            A_tile[row][col] = 0.0f;
        }

        if(col < M && row + phase*TILE < K) {
            B_tile[row][col] = B[(row + phase*TILE) * M + col];
        } else {
            B_tile[row][col] = 0.0f;
        }   
        __syncthreads();

        for(int k = 0; k < TILE; ++k) {
            value += A_tile[row][col+k] * B_tile[row+k][col];
        }
        __syncthreads();
    }
    C[row * M + col] = value;
}