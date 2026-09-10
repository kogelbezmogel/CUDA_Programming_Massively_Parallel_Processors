
#include "convolution2D_simple.cuh"

#define FILTER_R 1
#define BLOCK 32


void __global__ simple_convolution2D(float *A, float *F, float *B, int N, int M, int r) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float val = 0;
    for(int i = 0; i < 2*r+1; ++i) {
        for(int j = 0; j < 2*r+1; ++j) {
            if(col-r+j >= 0 && col-r+j < M && row-r+i >= 0 && row-r+i < N) {
                val += F[i*(2*r+1) + j] * A[(row-r+i) * M + (col-r+j)];
            }
        }
    }
    if(row < N && col < M) {
        B[row*M + col] = val;
    }
}