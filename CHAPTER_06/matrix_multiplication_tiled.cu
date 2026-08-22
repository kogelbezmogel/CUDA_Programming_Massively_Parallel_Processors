#include "matrix_multiplication_tiled.cuh"

#include <iostream>
#include "stdio.h"

#define TILE 32

void __global__ matrix_multiplication_tiled(float *A, float *B, float *C, int N, int K, int M) {
    float __shared__ A_tile[TILE][TILE];
    float __shared__ B_tile[TILE][TILE];
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float value = 0.0f;

    for(int phase = 0; phase < (K + TILE - 1) / TILE; ++phase) {

        if(row < N && threadIdx.x + phase*TILE < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + threadIdx.x + phase*TILE];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(col < M && threadIdx.y + phase*TILE < K) {
            B_tile[threadIdx.y][threadIdx.x] = B[(threadIdx.y + phase*TILE) * M + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }   
        __syncthreads();

        for(int k = 0; k < TILE; ++k) {
            value += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < N && col < M) {
        C[row * M + col] = value;
    }
}