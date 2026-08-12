#define _USE_MATH_DEFINES

#include "../UTILS/utils.h"

#include <iostream>
#include <stdio.h>
#include <cmath>

#define FILTER_R 1
#define BLOCK 32


void __host__ __device__ print_matrix(float *A, int N, int M);
void fill_matrix_with_gaussian_dist(float *A, int r);
void fill_matrix_with_data(float *A, int N, int M, uti::IntGenerator gen);
void __global__ simple_convolution2D(float *A, float *F, float *B, int N, int M, int r);


int main() {

    int N = 2048;
    int M = 2048;
    float *A_h, *B_h, *A_d, *B_d, *F_h, *F_d, *Dummy;
    float mseconds;
    
    cudaEvent_t start, stop;
    uti::IntGenerator gen(0, 10);

    int device_id = 0;
    int l2_size;
    cudaGetDevice(&device_id); // ?
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device_id);
    cudaMalloc((void**) &Dummy, 2 * l2_size * sizeof(float));
    cudaMemset((void *) Dummy, 0, 2 * l2_size * sizeof(float));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    A_h = new float[N*M];
    B_h = new float[N*M];
    F_h = new float[(2*FILTER_R+1) * (2*FILTER_R+1)];

    dim3 block_size(BLOCK, BLOCK, 1);
    dim3 grid_size((M + BLOCK-1) / BLOCK, (N + BLOCK-1) / BLOCK, 1);

    for(int i = 0; i < 50; ++i) {

        // populating input matrix
        fill_matrix_with_data(A_h, N, M, gen);

        //Filling filter with gaussian distribution.
        fill_matrix_with_gaussian_dist(F_h, FILTER_R);

        cudaMalloc((void**) &A_d, N * M * sizeof(float));
        cudaMalloc((void**) &B_d, N * M * sizeof(float));
        cudaMalloc((void**) &F_d, (2*FILTER_R+1) * (2*FILTER_R+1) * sizeof(float));
        cudaMemcpy(A_d, A_h, N * M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(F_d, F_h, (2*FILTER_R+1) * (2*FILTER_R+1) * sizeof(float) , cudaMemcpyHostToDevice);

        //flushing l2 cache
        cudaMemsetAsync((void*) Dummy, 0, 2 * l2_size * sizeof(float));
        cudaDeviceSynchronize();


        cudaEventRecord(start);
        simple_convolution2D<<<grid_size, block_size>>>(A_d, F_d, B_d, N, M, FILTER_R);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);

        cudaMemcpy(B_h, B_d, N*M*sizeof(float), cudaMemcpyDeviceToHost);

        // printf("\nFilter:\n");
        // print_matrix(F_h, 2*FILTER_R+1, 2*FILTER_R+1);
        // printf("\nInput:\n");
        // print_matrix(A_h, N, M);
        // printf("\nResult:\n");
        // print_matrix(B_h, N, M);

        cudaEventElapsedTime(&mseconds, start, stop);
        printf("Time elasped: %f seconds\n", mseconds/1000);
    }

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


void fill_matrix_with_data(float *A, int N, int M, uti::IntGenerator gen) {
    for(int i = 0; i < N*M; ++i) {
        A[i] = gen.next();
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