#include <iostream>
#include "stdio.h"

#include "matrix_multiplication.cuh"

// implement 3 multiplication strategies.
// put each kernel in diffrent file
// make framework to estimate performance of those
// make some performance analysis. Collect data, draw plots
// review, clean, estimate perfomrance of following chapters


void __global__ matrix_multiplication(float *A, float *B, float *C, int N, int K, int M) {
    // simple multiplication kernel 1 thread computates one element from C matrix
    
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = 0.0f;
    if(row < N && col < M) {
        for(int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * M + col]; 
        }   
    }

    if(row < N && col < M) {
        A[row * M + col] = sum;
    }
}