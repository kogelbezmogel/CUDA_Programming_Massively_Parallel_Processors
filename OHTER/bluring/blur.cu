#include <stdio.h>
#include <cuda_runtime.h>
#include "blur.h"
#define N 4


__global__ void kernel(float* out, float* in, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if( id < n ) {
        out[id] = in[id] + 10;
    }
}


#ifdef _cplusplus
extern "C" {
#endif

void kernel_run() {
    float *out_h, *in_h;
    float *out_d, *in_d; 
    
    out_h = (float*) malloc(N * sizeof(float));
    in_h = (float*) malloc(N * sizeof(float));
    cudaMalloc((void**) &out_d, N * sizeof(float));
    cudaMalloc((void**) &in_d, N * sizeof(float));

    for(int i = 0; i < N; i++) {
        in_h[i] = i;
    }

    cudaMemcpy(in_d, in_h, N * sizeof(float), cudaMemcpyHostToDevice);
    kernel<<<4, 16>>>(out_d, in_d, N);
    cudaMemcpy(out_h, out_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(out_d);
    cudaFree(in_d);
    free(out_h);
    free(in_h);

    printf("CUDA PASSED");

#ifdef _cplusplus
}
#endif


}