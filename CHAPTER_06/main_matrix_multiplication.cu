#define BLOCK 32

#include "matrix_multiplication.cuh"
#include "matrix_multiplication_tiled.cuh"


// here will be whole perfomance analysis code 

int main() {

    int N = 4;
    int K = 4;
    int M = 4;

    float *A_h, *B_h, *C_h, *A_d, *B_d, *C_d;

    A_h = new float[N * K];
    B_h = new float[K * M];
    C_h = new float[N * M];

    cudaMalloc((void**) &A_d, N * K * sizeof(float));
    cudaMalloc((void**) &B_d, K * M * sizeof(float));
    cudaMalloc((void**) &C_d, N * M * sizeof(float));

    cudaMemcpy(A_d, A_h, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_size((M+BLOCK-1) / BLOCK, (N+BLOCK-1) / BLOCK, 1);
    dim3 block_size(BLOCK, BLOCK, 1);
    
    matrix_multiplication<<<block_size, grid_size>>>(A_d, B_d, C_d, N, K, M);

    cudaMemcpy(C_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    delete A_h;
    delete B_h;
    delete C_h;

    return 0;
}