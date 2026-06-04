#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 4

__global__ void scan_kogge_stone_mblocks(float *A, float *B, float *S, int N);
__global__ void scan_kogge_stone_sblock(float *B, float *S, int N);


int main() {
    float *A_h, *A_d, *B_h, *B_d, *S_h, *S_d;
    int N = 12;
    A_h = new float[N];
    B_h = new float[N];
    S_h = new float[(N + BLOCK-1) / BLOCK];

    cudaMalloc((void**) &A_d, sizeof(float) * N);
    cudaMalloc((void**) &B_d, sizeof(float) * N);
    cudaMalloc((void**) &S_d, sizeof(float) * (N + BLOCK-1) / BLOCK);

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) { A_h[i] = generator(); }

    for(int i = 0; i < N; ++i) {
        if(i % BLOCK == 0) { printf("| "); }
        printf("%5.0f ", A_h[i]);
    }
    printf("\n");

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size( (N + BLOCK - 1) / BLOCK );
    dim3 block_size(BLOCK);

    scan_kogge_stone_mblocks<<<grid_size, block_size>>>(A_d, B_d, S_d, N);

    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i) {
        if(i % BLOCK == 0) { printf("| "); }
        printf("%5.0f ", B_h[i]);
    }
    printf("\n\n");

    cudaMemcpy(S_h, S_d, sizeof(float) * (N + BLOCK-1) / BLOCK, cudaMemcpyDeviceToHost);
    for(int i = 0; i < (N + BLOCK-1) / BLOCK; ++i) { printf("%5.0f ", S_h[i]); }
    printf("\n\n\n");

    scan_kogge_stone_sblock<<<1, block_size>>>(B_d, S_d, N);

    cudaMemcpy(S_h, S_d, sizeof(float) * (N + BLOCK-1) / BLOCK, cudaMemcpyDeviceToHost);
    for(int i = 0; i < (N + BLOCK-1) / BLOCK; ++i) { printf("%5.0f ", S_h[i]); }
    printf("\n\n");

    cudaMemcpy(B_h, B_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; ++i) {
        if(i % BLOCK == 0) { printf("| "); }
        printf("%5.0f ", B_h[i]);
    }
    printf("\n\n");

    delete [] B_h;
    delete [] A_h;
    cudaFree(A_d);
    cudaFree(B_d);

    return 0;
}


__global__ void scan_kogge_stone_mblocks(float *A, float *B, float *S, int N) {
    /*  This kernel takes arbitrary long input A and computes kogge stone in each block separatly.
        Results are saved to B array. Additionaly it stores last values of each block in S array for second phase.
    */
    __shared__ float B_sh[BLOCK];
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Loading data to shared memory
    if(id < N) {
        B_sh[threadIdx.x] = A[id];
    } else {
        B_sh[threadIdx.x] = 0.0f;
    }

    float temp;
    for(int stride = 1; stride < BLOCK; stride *= 2) {
        __syncthreads();
        if(threadIdx.x >= stride) {
            temp = B_sh[threadIdx.x] + B_sh[threadIdx.x - stride]; 
        }

        __syncthreads();
        if(threadIdx.x >= stride) {
            B_sh[threadIdx.x] = temp;
        }
    }

    __syncthreads();
    if(id < N) {
        B[id] = B_sh[threadIdx.x];
    }
    
    if(threadIdx.x == blockDim.x-1 && id < N) {
        S[blockIdx.x] = B_sh[threadIdx.x];
    } else if(id == N) {
        S[blockIdx.x] = B_sh[threadIdx.x];
    }

}

__global__ void scan_kogge_stone_sblock(float *B, float *S, int N) {
/*  This kernel takes array S and performs kogge stone with single block on it. At the end it adds 
    results of S scan to results stored in B array what finishes the algorithm.
*/
    __shared__ float S_sh[BLOCK];

    // Loading S to shared memory
    if(threadIdx.x < blockDim.x) {
        S_sh[threadIdx.x] = S[threadIdx.x];
    } else {
        S_sh[threadIdx.x] = 0.0f;
    }

    // Performing kogge_stone with single block on S
    float temp;
    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if(threadIdx.x >= stride && threadIdx.x < blockDim.x) {
            temp = S_sh[threadIdx.x] + S_sh[threadIdx.x - stride];
        }

        __syncthreads();
        if(threadIdx.x >= stride && threadIdx.x < blockDim.x) {
            S_sh[threadIdx.x] = temp;
        }
    }

    // Adding results from S_sh on the partial sums in B
    __syncthreads();
    if(threadIdx.x > 0 && threadIdx.x < blockDim.x) {
        for(int i = 0; i < BLOCK; ++i) {
            if(threadIdx.x * BLOCK + i < N) {
                B[threadIdx.x * BLOCK + i] += S_sh[threadIdx.x-1];
            }

        }
    }

    // Updating global S just for giggles
    if(threadIdx.x < blockDim.x) {
        S[threadIdx.x] = S_sh[threadIdx.x];
    }

}
