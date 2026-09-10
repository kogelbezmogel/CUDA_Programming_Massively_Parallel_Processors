#include "convolution2D_tiled_v1.cuh"

#define FILTER_R 1
#define IN_TILE 32
#define OUT_TILE ((IN_TILE) - 2*(FILTER_R))


__constant__ float F[2*FILTER_R+1][2*FILTER_R+1];


void __global__ convolution2D_tiled_v1(float *A, float *B, int N, int M, int r) {
    int col = blockIdx.x * OUT_TILE + threadIdx.x - r;
    int row = blockIdx.y * OUT_TILE + threadIdx.y - r;

    __shared__ float S_mem[IN_TILE][IN_TILE];

    if(col >= 0 && col < M && row >= 0 && row < N) {
        S_mem[threadIdx.y][threadIdx.x] = A[row*M + col];
    } else {
        S_mem[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();


    float val = 0;
    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    if(row >= 0 && row < N && col >= 0 && col < M) {
        if(local_row >= r && local_row < OUT_TILE+r && local_col >= r && local_col < OUT_TILE+r) {
            for(int i = 0; i < 2*r+1; ++i) {
                for(int j = 0; j < 2*r+1; ++j) {
                    val += F[i][j] * S_mem[local_row-r+i][local_col-r+j];
                }
            }
            B[row * M + col] = val;
        }
    }
}