#define _USE_MATH_DEFINES

#include <iostream>
#include <stdio.h>
#include <cmath>

#define FILTER_R 1
#define BLOCK 32


void __host__ __device__ print_matrix(float *A, int N, int M);
void fill_matrix_with_gaussian_dist(float *A, int r);
void fill_matrix_with_data(float *A, int N, int M, int seed = 967);
void __global__ simple_convolution2D(float *A, float *F, float *B, int N, int M, int r);


int main() {

    int N = 4;
    int M = 4;
    float *A_h, *B_h, *A_d, *B_d, *F_h, *F_d;

    A_h = new float[N*M];
    B_h = new float[N*M];
    F_h = new float[(2*FILTER_R+1) * (2*FILTER_R+1)];

    // populating input matrix
    fill_matrix_with_data(A_h, N, M);

    //Filling filter with gaussian distribution.
    fill_matrix_with_gaussian_dist(F_h, FILTER_R);

    cudaMalloc((void**) &A_d, N * M * sizeof(float));
    cudaMalloc((void**) &B_d, N * M * sizeof(float));
    cudaMalloc((void**) &F_d, (2*FILTER_R+1) * (2*FILTER_R+1) * sizeof(float));
    cudaMemcpy(A_d, A_h, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, (2*FILTER_R+1) * (2*FILTER_R+1) * sizeof(float) , cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK, BLOCK, 1);
    dim3 grid_size((M + BLOCK-1) / BLOCK, (N + BLOCK-1) / BLOCK, 1);
    simple_convolution2D<<<grid_size, block_size>>>(A_d, F_d, B_d, N, M, FILTER_R);
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