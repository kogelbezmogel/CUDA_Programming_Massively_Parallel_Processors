/*
Second version of tiled convolution uses tiles with smaller size.
In result every thread computes convolution but also each thread
needs to load more than 1 element from global memory what must be optimized.
*/

#define _USE_MATH_DEFINES

#include <iostream>
#include <stdio.h>
#include <cmath>

#define FILTER_R 1
#define TILE_SIZE 32

__constant__ float F[2*FILTER_R+1][2*FILTER_R+1];


void __host__ __device__ print_matrix(float *A, int N, int M);
void fill_matrix_with_gaussian_dist(float *A, int r);
void fill_matrix_with_data(float *A, int N, int M, int seed = 967);


void __global__ convolution2D_tiled(float *A, float *B, int N, int M, int r) {
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


int main() {

    int N = 4;
    int M = 4;
    float *A_h, *B_h, *A_d, *B_d, *F_h;

    A_h = new float[N*M];
    B_h = new float[N*M];
    F_h = new float[(2*FILTER_R+1) * (2*FILTER_R+1)];

    // populating input matrix
    fill_matrix_with_data(A_h, N, M);

    //Filling filter with gaussian distribution.
    fill_matrix_with_gaussian_dist(F_h, FILTER_R);

    cudaMalloc((void**) &A_d, N * M * sizeof(float));
    cudaMalloc((void**) &B_d, N * M * sizeof(float));
    cudaMemcpy(A_d, A_h, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, F_h, (2*FILTER_R+1) * (2*FILTER_R+1) * sizeof(float));


    dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_size((M+TILE_SIZE-1) / TILE_SIZE, (N+TILE_SIZE-1) / TILE_SIZE, 1);
    convolution2D_tiled<<<grid_size, block_size>>>(A_d, B_d, N, M, FILTER_R);
    

    cudaMemcpy(B_h, B_d, N*M*sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nFilter:\n");
    print_matrix(F_h, 2*FILTER_R+1, 2*FILTER_R+1);
    printf("\nInput:\n");
    print_matrix(A_h, N, M);
    printf("\nResult:\n");
    print_matrix(B_h, N, M);
    
    std::cout << "\nFINISHED\n";
    return 0;
}


void __host__ __device__ print_matrix(float *A, int N, int M) {
    for(int i = 0; i < N; ++i) {
        printf("[");
        for(int j = 0; j < M-1; ++j) {
            printf("%4.1f ", A[i*M + j]);
        }
        printf("%4.1f ]\n", A[i*M + M-1]);
    }
}


void fill_matrix_with_gaussian_dist(float *A, int r) {
    float coeff = 1.0 / (std::sqrt(2 * M_PI));
    for(int i = -r; i < r+1; ++i) {
        for(int j = -r; j < r+1; ++j) {
            A[(i+r) * (2*r+1) + (j+r)] = coeff * std::exp( (-0.5) * (i*i + j*j));
        }
    }
}


void fill_matrix_with_data(float *A, int N, int M, int seed) {
    int num = seed;
    for(int i = 0; i < N*M; ++i) {
        num = (1091 * num + 1093) % 101;
        A[i] = num % 10;
    }
}