#include <iostream>
#include <assert.h>
#include <stdio.h>
#include "multiply_std.h"

void matrix_row2col_layout(float* A_col_major, float* A_row_major, int N, int M);

void __host__ __device__ print_matrix(float *A, int N, int M);

void __global__ kernel_multiply_matrices(float *A, float *B, float *C, int N, int K, int M, int TAIL_SIZE);

void __global__ kernel_multiply_matrices_with_column_layout(float *A, float *B, float *C, int N, int K, int M, int TAIL_SIZE);


int main() {

    float *A_h, *B_h, *C_h, *B_h_col_major, *C_e;
    float *A_d, *B_d, *C_d, *B_d_col_major;

    int N = 3;
    int K = 4;
    int M = 5;
    int TAIL_SIZE = 2;

    A_h = new float[N * K];
    B_h = new float[K * M];
    C_h = new float[N * M];
    C_e = new float[N * M];
   
    int seed = 967;
    int num = seed;
    for(int i = 0; i < N*K; ++i) {
        num = (1091 * num + 1093) % 101;
        A_h[i] = num % 10;
    }   
    num = seed;
    for(int i = 0; i < K*M; ++i) {
        num = (1381 * num + 1499) % 101;   
        B_h[i] = num % 10;
    }

    // Computing expected result for matrix multiplycation
    multiply_matrices_cpu(A_h, B_h, C_e, N, K, M);

    // Preparing resources for device
    cudaMalloc((void**) &A_d, sizeof(float) * N*K);
    cudaMalloc((void**) &B_d, sizeof(float) * K*M);
    cudaMalloc((void**) &C_d, sizeof(float) * N*M);

    cudaMemcpy(A_d, A_h, sizeof(float)*N*K, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*K*M, cudaMemcpyHostToDevice);

    // evaluating grid size and shared memory
    dim3 block_size(TAIL_SIZE, TAIL_SIZE, 1);
    dim3 grid_size((M+TAIL_SIZE-1) / TAIL_SIZE, (N+TAIL_SIZE-1) / TAIL_SIZE, 1);
    size_t shmem_size = 2 * TAIL_SIZE * TAIL_SIZE * sizeof(float);

    // Standard kernel start
    kernel_multiply_matrices<<<grid_size, block_size, shmem_size>>>(A_d, B_d, C_d, N, K, M, TAIL_SIZE);

    // Copying kernel results
    cudaMemcpy(C_h, C_d, sizeof(float)*N*M, cudaMemcpyDeviceToHost);

    // Evaluating the results for standard execution
    for(int i = 0; i < N*M; ++i) {
        assert(C_h[i] - C_e[i] < 1e-4);
    }

    // Preparing resources for another kernel start
    cudaMalloc((void**) &B_d_col_major, sizeof(float)*K*M);

    B_h_col_major = new float[K * M];
    matrix_row2col_layout(B_h_col_major, B_h, K, M);

    cudaMemcpy(B_d_col_major, B_h_col_major, sizeof(float)*N*K, cudaMemcpyHostToDevice);

    // Kernel for changed layout start
    kernel_multiply_matrices_with_column_layout<<<grid_size, block_size, shmem_size>>>(A_d, B_d_col_major, C_d, N, K, M, TAIL_SIZE);

    cudaMemcpy(C_h, C_d, sizeof(float)*N*M, cudaMemcpyDeviceToHost);

    // Evaluating results for execution with changed layout
    for(int i = 0; i < N*M; ++i) {
        assert(C_h[i] - C_e[i] < 1e-4);
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(B_d_col_major);
    delete [] A_h;
    delete [] B_h;
    delete [] C_h;
    delete [] C_e;
    delete [] B_h_col_major;


    std::cout << "PASSED\n";
    return 0;
}

void matrix_row2col_layout(float* A_col_major, float* A_row_major, int N, int M) {
    int col, row;
    
    for(int i = 0; i < N*M; ++i) {
        row = i / M;
        col = i % M;
        
        A_col_major[col * N + row] = A_row_major[i];
    }
}


void __host__ __device__ print_matrix(float *A, int N, int M) {
    for(int i = 0; i < N; ++i) {
        printf("[");
        for(int j = 0; j < M-1; ++j) {
            printf("%3.0f ", A[i*M + j]);
        }
        printf("%3.0f]\n", A[i*M + M-1]);
    }
}

void __global__ kernel_multiply_matrices(float *A, float *B, float *C, int N, int K, int M, int TAIL_SIZE) {
    // This is standard kernel multiply from previus chapters. It uses 
    // memory burst naturally while loading tails.

    // Allocated dynamic shared memory
    extern __shared__ char A_and_B_tail[];

    // Creating pointer for easier acces to shared memory
    float* A_tail = (float*) A_and_B_tail;
    float* B_tail = (float*) (A_and_B_tail + TAIL_SIZE * TAIL_SIZE * sizeof(float));

    // Identification of thread which coresponds to row and col of output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Variable to accumulate multiplication across all steps
    float mul_sum = 0;
    int num_of_steps = (K + TAIL_SIZE - 1) / TAIL_SIZE;
    for(int step = 0; step < num_of_steps; ++step) {

        // Loading Tails from global memory
        if(row < N && threadIdx.x + step*TAIL_SIZE < K) {
            A_tail[threadIdx.y * TAIL_SIZE + threadIdx.x] = A[row*K + threadIdx.x + step*TAIL_SIZE];
        } else {
            A_tail[threadIdx.y * TAIL_SIZE + threadIdx.x] = 0;
        }

        if(col < M && threadIdx.y + step*TAIL_SIZE < K) {
            B_tail[threadIdx.y * TAIL_SIZE + threadIdx.x] = B[(threadIdx.y + step*TAIL_SIZE) * M + col];
        } else {
            B_tail[threadIdx.y * TAIL_SIZE + threadIdx.x] = 0;
        }
        __syncthreads();

        // Accumulating multiplications
        for(int j = 0; j < TAIL_SIZE; ++j) {
            mul_sum += A_tail[threadIdx.y * TAIL_SIZE + j] * B_tail[j * TAIL_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    // Saving result to output matrix
    if(row < N && col < M) {
        C[row*M + col] = mul_sum;
    }
}


void __global__ kernel_multiply_matrices_with_column_layout(float *A, float *B, float *C, int N, int K, int M, int TAIL_SIZE) {
    // In this scenarion only matrix B is stored in column major layout.
    // To speed up multiplication by usage of memory burst in this algorithm 
    // technique of corner turning will be used.

    extern __shared__ char A_and_B_tail[];

    float *A_tail = (float*) A_and_B_tail;
    float *B_tail = (float*) A_and_B_tail + TAIL_SIZE * TAIL_SIZE * sizeof(float);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float mul_sum = 0;
    int num_of_steps = (K + TAIL_SIZE - 1) / TAIL_SIZE;
    int len = 0;

    for(int step = 0; step < num_of_steps; ++step) {

        len = threadIdx.y * TAIL_SIZE + threadIdx.x;

        if(row < N && threadIdx.x + step*TAIL_SIZE < K) {
            A_tail[len] = A[row * K + threadIdx.x + step*TAIL_SIZE];
        } else {
            A_tail[len] = 0;
        }

        // The idea is that len-th thread in block need to read len-th element in slice of B
        // but elements in B are ordered in column-major layout so the slice needs to be read column wise.

        // SLICE of B 2x3
        //                  |0|2|4|
        //                  |1|3|5|

        // BLOCK 2x3        
        //                  |0|1|2|
        //                  |3|4|5|

        // In the begging we want to find first element in B coresponding to first thread in block
        // and then move len/T columns right and len%T rows down to find len-th element in slice to read.
        // First element in B is at (col - thx) column and in step*T row
        // At the end read element from B must be saved to its original column and row 
        if(step*TAIL_SIZE + len%TAIL_SIZE < K && col - threadIdx.x + len/TAIL_SIZE < M) {
            B_tail[len%TAIL_SIZE * TAIL_SIZE + len/TAIL_SIZE] = B[(col - threadIdx.x + len/TAIL_SIZE) * K + step * TAIL_SIZE + len%TAIL_SIZE];
        } else {
            B_tail[len%TAIL_SIZE * TAIL_SIZE + len/TAIL_SIZE] = 0;
        }

        for(int k = 0; k < TAIL_SIZE; ++k) {
            mul_sum += A_tail[threadIdx.y * TAIL_SIZE + k] * B_tail[k * TAIL_SIZE + threadIdx.x];
        }
        __syncthreads();
    }
    if(row < N && col < M)
        C[row * M + col] = mul_sum;
}
