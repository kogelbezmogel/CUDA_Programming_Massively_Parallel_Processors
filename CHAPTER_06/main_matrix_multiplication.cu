#define BLOCK 4

#include "../UTILS/utils.h"
#include "../UTILS/matrix_multiplication.h"
#include "matrix_multiplication.cuh"
#include "matrix_multiplication_tiled.cuh"
#include "matrix_multiplication_coarsed.cuh"

// #include <vector>
// #include <unordered_map>


// here will be whole perfomance analysis code 

int main() {

    int device = 0;
    int regis_per_block;
    int shmem_per_block;
    int threa_per_block;
    int regis_per_sm;
    int shmem_per_sm;
    int threa_per_sm;
    int warp_size;

    cudaDeviceGetAttribute(&regis_per_block, cudaDevAttrMaxRegistersPerBlock, device);
    cudaDeviceGetAttribute(&shmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&threa_per_block, cudaDevAttrMaxThreadsPerBlock, device);
    cudaDeviceGetAttribute(&regis_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, device);
    cudaDeviceGetAttribute(&shmem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    cudaDeviceGetAttribute(&threa_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device);

    printf("GPU ------------------------\n");
    printf("Max registers per block:     %6d floats\n", regis_per_block);
    printf("Max registers per sm:        %6d floats\n", regis_per_sm);
    printf("Max shared memory per block: %6d B = %5lu floats\n", shmem_per_block, shmem_per_block / sizeof(float));
    printf("Max shared memory per sm:    %6d B = %5lu floats\n", shmem_per_sm, shmem_per_sm / sizeof(float));
    printf("Max threads per block:       %6d threads\n", threa_per_block);
    printf("Max threads per sm:          %6d threads\n", threa_per_sm);
    printf("Warp size:                   %6d threads\n", warp_size);
    
    // int N = 8;
    // int K = 8;
    // int M = 8;

    // float *A_h, *B_h, *C_h, *A_d, *B_d, *C_d, *C_t;

    // A_h = new float[N * K];
    // B_h = new float[K * M];
    // C_h = new float[N * M];
    // C_t = new float[N * M];

    // cudaMalloc((void**) &A_d, N * K * sizeof(float));
    // cudaMalloc((void**) &B_d, K * M * sizeof(float));
    // cudaMalloc((void**) &C_d, N * M * sizeof(float));

    // uti::IntGenerator gen(0, 10);
    // uti::fill_memory_with_random_data(A_h, N*K, gen);
    // uti::fill_memory_with_random_data(B_h, K*M, gen);

    // matrix_multiplication_cpu(A_h, B_h, C_t, N, K, M);

    // cudaMemcpy(A_d, A_h, N * K * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(B_d, B_h, K * M * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 grid_size((M+BLOCK-1) / BLOCK, (N+BLOCK-1) / BLOCK, 1);
    // dim3 block_size(BLOCK, BLOCK, 1);
    
    // matrix_multiplication_coarsed<<<grid_size, block_size>>>(A_d, B_d, C_d, N, K, M);

    // cudaMemcpy(C_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // uti::check_abs_error(C_h, C_t, N*M);

    // delete A_h;
    // delete B_h;
    // delete C_h;
    // cudaFree(A_d);
    // cudaFree(B_d);
    // cudaFree(C_d);

    return 0;
}