/*
Second version of tiled convolution uses tiles with smaller size.
In result every thread computes convolution but also each thread
needs to load more than 1 element from global memory what must be optimized.
*/

#include "convolution2D_tiled_v2.cuh"


#define FILTER_R 1
#define IN_TILE 32
#define TILE_SIZE 32
#define OUT_TILE ((IN_TILE) - 2*(FILTER_R))


__constant__ float F[2*FILTER_R+1][2*FILTER_R+1];



void __global__ convolution2D_tiled_v2(float *A, float *B, int N, int M, int r) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_sh[(TILE_SIZE + 2*FILTER_R) * (TILE_SIZE + 2*FILTER_R)];

    // loading data from global memory
    // to optimize this preocess with memory burst 
    // consecutive threads need to load consecutive elements
    int slice_start_global_row = blockIdx.y * blockDim.y - r;
    int slice_start_global_col = blockIdx.x * blockDim.x - r;
    for(int ind = threadIdx.y * blockDim.x + threadIdx.x; ind < (TILE_SIZE+2*r) * (TILE_SIZE+2*r); ind += TILE_SIZE * TILE_SIZE) {
        int element_global_row = slice_start_global_row + ind / (TILE_SIZE+2*r);
        int element_global_col = slice_start_global_col + ind % (TILE_SIZE+2*r);
        
        if(element_global_row >= 0 && element_global_row < N && element_global_col >= 0 && element_global_col < M) {
            A_sh[ind] = A[element_global_row * M + element_global_col];
        } else {

            A_sh[ind] = 0.0f;
        }
    }
    __syncthreads();

    // computing convolution
    float val = 0.0f;
    if(row < N && col < M) {
        for(int i = 0; i < 2*r+1; ++i) {
            for(int j = 0; j < 2*r+1; ++j) {
                val += A_sh[(threadIdx.y+i) * (TILE_SIZE+2*r) + threadIdx.x+j] * F[i][j];
            }
        }
        B[row*M + col] = val;
    }
}