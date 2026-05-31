#include <iostream>
#include <assert.h>
#include <math.h>
#include "multiply_std.h"


__global__ void multiply_vector_matrix(float *A, float *B, float *C, int N, int M) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float s_value = 0;

    if(id < M) {
        for(int i = 0; i < N; ++i){
            s_value += A[i] * B[i*M + id];
        }
        C[id] = s_value;
    }


}


int main() {

    int N = 30;
    int M = 40;

    float *A_h, *B_h, *C_h, *C_e;
    float *A_d, *B_d, *C_d;

    A_h = new float[N];
    B_h = new float[N * M];
    C_h = new float[M];
    C_e = new float[M];

    for(int i = 0; i < N; ++i) 
        A_h[i] = (i * i * i + 5) % 7;
        
    for(int i = 0; i < N * M; ++i) 
        B_h[i] = (i * i * i + 5) % 7;

    multiply_vector_matrix_cpu(A_h, B_h, C_e, N, M);

    cudaMalloc((void**) &A_d, N * sizeof(float));
    cudaMalloc((void**) &B_d, N * M * sizeof(float));
    cudaMalloc((void**) &C_d, M * sizeof(float));

    cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * M * sizeof(float), cudaMemcpyHostToDevice);

    multiply_vector_matrix<<<ceil((float) M / 256), 256>>>(A_d, B_d, C_d, N, M);
    cudaMemcpy(C_h, C_d, M * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < M; ++i)
        assert(C_e[i] - C_h[i] < 1e-4);

    std::cout << "PASSED\n";
    return 0;
}
