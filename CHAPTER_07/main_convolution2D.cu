#define _USE_MATH_DEFINES

#include <iostream>
#include <stdio.h>
#include <cmath>

#include "convolution2D_const_memory.cuh"
#include "convolution2D_tiled_v2.cuh"
#include "convolution2D_tiled_v1.cuh"
#include "convolution2D_simple.cuh"
#include "../UTILS/utils.h"

#define FILTER_R 1
#define BLOCK 32


int main() {

    int N = 4;
    int M = 4;
    float *A_h, *B_h, *A_d, *B_d, *F_h, *F_d;

    A_h = new float[N*M];
    B_h = new float[N*M];
    F_h = new float[(2*FILTER_R+1) * (2*FILTER_R+1)];

    uti::IntGenerator gen(0, 10);

    // populating input matrix
    uti::fill_memory_with_random_data(A_h, N*M, gen);

    //Filling filter with gaussian distribution.
    uti::fill_tensor_with_gaussian_dist(F_h, FILTER_R);

    cudaMalloc((void**) &A_d, N * M * sizeof(float));
    cudaMalloc((void**) &B_d, N * M * sizeof(float));
    cudaMalloc((void**) &F_d, (2*FILTER_R+1) * (2*FILTER_R+1) * sizeof(float));
    cudaMemcpy(A_d, A_h, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, (2*FILTER_R+1) * (2*FILTER_R+1) * sizeof(float) , cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK, BLOCK, 1);
    dim3 grid_size((M + BLOCK-1) / BLOCK, (N + BLOCK-1) / BLOCK, 1);
    simple_convolution2D<<<grid_size, block_size>>>(A_d, F_d, B_d, N, M, FILTER_R);
    cudaMemcpy(B_h, B_d, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nFINISHED\n";
    return 0;
}