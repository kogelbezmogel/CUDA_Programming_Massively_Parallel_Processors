#define _USE_MATH_DEFINES

#include <iostream>
#include <stdio.h>
#include <cmath>

#define FILTER_R 1
#define IN_TILE 32
#define OUT_TILE ((IN_TILE) - 2*(FILTER_R))


__constant__ float F[2*FILTER_R+1][2*FILTER_R+1];


void __global__ convolution2D_tiled(float *A, float *B, int N, int M, int r) {
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


void __host__ __device__ print_matrix(float *A, int N, int M);
void fill_matrix_with_gaussian_dist(float *A, int r);
void fill_matrix_with_data(float *A, int N, int M, int seed = 967);


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


    dim3 block_size(IN_TILE, IN_TILE, 1);
    dim3 grid_size((M+OUT_TILE-1) / OUT_TILE, (N+OUT_TILE-1) / OUT_TILE, 1);
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