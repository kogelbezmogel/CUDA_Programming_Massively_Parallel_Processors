#include <iostream>
#include <stdio.h>
#include "../CHAPTER_09/RandomIntGenerator.h"

#define BLOCK 32


__global__ void scan_brent_kung(float *A, int N);


int main() {
    float *A_h, *A_d;
    int N = 12;
    A_h = new float[N];

    cudaMalloc((void**) &A_d, sizeof(float) * N);

    RandomIntGenerator generator(967, 0, 10);

    for(int i = 0; i < N; ++i) { A_h[i] = generator(); }

    cudaMemcpy(A_d, A_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 grid_size( (N + BLOCK - 1) / BLOCK );
    dim3 block_size(BLOCK);
    scan_brent_kung<<<grid_size, block_size>>>(A_d, N);
    
    cudaMemcpy(A_h, A_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    delete [] A_h;
    cudaFree(A_d);

    return 0;
}


__global__ void scan_brent_kung(float *A, int N) {

}

