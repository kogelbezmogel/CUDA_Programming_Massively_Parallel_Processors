#define _USE_MATH_DEFINES

#include <iostream>
#include <stdio.h>
#include <cmath>

#define FILTER_R 1
#define BLOCK 8


__constant__ float F[2*FILTER_R+1][2*FILTER_R+1][2*FILTER_R+1];


void __global__ convolution3D_simple(float *A, float *B, int N, int K, int M, int r);

void __host__ __device__ print_tensor(float *A, int N, int K, int M);

void fill_tensor_test(float *A, int r);

void fill_tensor_with_data(float *A, int N, int K, int M, int seed = 967);

void fill_tensor_with_gaussian_dist(float *A, int r);

void fill_tensor_with_test_data(float *A, int N, int K, int M);


int main() {

    int N = 3;
    int M = 3;
    int K = 3;
    float *A_h, *B_h, *A_d, *B_d, *F_h;

    A_h = new float[N*K*M];
    B_h = new float[N*K*M];
    F_h = new float[(2*FILTER_R+1) * (2*FILTER_R+1) * (2*FILTER_R+1)];

    // populating input matrix
    fill_tensor_with_data(A_h, N, K, M);

    //Filling filter with gaussian distribution.
    fill_tensor_test(F_h, FILTER_R);

    cudaMalloc((void**) &A_d, N * K * M * sizeof(float));
    cudaMalloc((void**) &B_d, N * K * M * sizeof(float));
    cudaMemcpy(A_d, A_h, N * K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, F_h, (2*FILTER_R+1) * (2*FILTER_R+1) * (2*FILTER_R+1) * sizeof(float));

    dim3 block_size(BLOCK, BLOCK, BLOCK);
    dim3 grid_size((K+BLOCK-1) / BLOCK, (N+BLOCK-1) / BLOCK, (M+BLOCK-1) / BLOCK);
    convolution3D_simple<<<grid_size, block_size>>>(A_d, B_d, N, K, M, FILTER_R);
    cudaMemcpy(B_h, B_d, N * K * M * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nFilter:\n");
    print_tensor(F_h, 2*FILTER_R+1, 2*FILTER_R+1, 2*FILTER_R+1);
    printf("\nInput:\n");
    print_tensor(A_h, N, K, M);
    printf("\nResult:\n");
    print_tensor(B_h, N, K, M);
    
    std::cout << "\nFINISHED\n";
    return 0;
}



void __global__ convolution3D_simple(float *A, float *B, int N, int K, int M, int r) {
    
    int pla = blockIdx.z * blockDim.z + threadIdx.z;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float val = 0;
    for(int m = 0; m < 2*r+1; ++m) {
        for(int n = 0; n < 2*r+1; ++n) {
            for(int k = 0; k < 2*r+1; ++k) {
                if(pla-r+m >= 0 && pla-r+m < M && row-r+n >= 0 && row-r+n < N && col-r+k >= 0 && col-r+k < K) {
                   val += F[m][n][k] * A[(pla-r+m)*N*K + (row-r+n)*K + col-r+k]; 
                }
            }
        }
    }

    if(pla < M && row < N && col < K) {
        B[pla*N*K + row*K + col] = val;
    }
}


void __host__ __device__ print_tensor(float *A, int N, int K, int M) {
    for(int m = 0; m < M; ++m) {
        printf("[\n");
        for(int n = 0; n < N; ++n) {
            printf("    [");
            for(int k = 0; k < K-1; ++k) {
                printf("%4.1f ", A[m*K*N + n*K + k]);
            }
            printf("%4.1f ]\n", A[m*K*N + n*K + K-1]);
        }
        printf("]\n");
    }
}


void fill_tensor_with_gaussian_dist(float *A, int r) {
    float coeff = 1.0 / (std::sqrt(2 * M_PI));
    for(int i = -r; i < r+1; ++i) {
        for(int j = -r; j < r+1; ++j) {
            for(int k = -r; k < r+1; ++k){
                A[(i+r)*(2*r+1)*(2*r+1) + (j+r)*(2*r+1) + k+r] = coeff * std::exp( (-0.33) * (i*i + j*j + k*k));
            }
        }
    }
}


void fill_tensor_test(float *A, int r) {
    for(int i = 0; i < 2*r+1; ++i) {
        for(int j = 0; j < 2*r+1; ++j) {
            for(int k = 0; k < 2*r+1; ++k){
                A[i*(2*r+1)*(2*r+1) + j*(2*r+1) + k] = i*(2*r+1)*(2*r+1) + j*(2*r+1) + k;
            }
        }
    }
}


void fill_tensor_with_data(float *A, int N, int K, int M, int seed) {
    int num = seed;
    for(int i = 0; i < N*K*M; ++i) {
        num = (1091 * num + 1093) % 101;
        A[i] = num % 10;
    }
}


void fill_tensor_with_test_data(float *A, int N, int K, int M) {
    for(int i = 0; i < N*K*M; ++i) {
        A[i] = i;
    }
}