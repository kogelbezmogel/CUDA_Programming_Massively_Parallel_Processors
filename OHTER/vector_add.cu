#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#define N 10000000
#define M 256
#define T 256

#define MAX_ERR 1e-5


__global__ void vector_add(float* out, float* a, float* b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;

    // the method for thread to move along the data is a little counterinvtuitive
    // each thread is not moving along continous data segment but it uses elements 
    // spaced apart with given stride

    for(int j = index; j < n; j += stride) {
        out[j] = a[j] + b[j];
    }
}


int main() {
    float *out, *a, *b;
    float *d_out, *d_a, *d_b;

    a =     (float*) malloc(N * sizeof(float));
    b =     (float*) malloc(N * sizeof(float));
    out =   (float*) malloc(N * sizeof(float));
    
    for(int i = 0; i < N; i++) {
        a[i] = (float) (i % 7);
        b[i] = (float) 1.0f; 
    }

    // using & to take pointer to d_a which is a pointer and
    // then converting it float** -> void**
    cudaMalloc((void**) &d_a, N * sizeof(float));
    cudaMalloc((void**) &d_b, N * sizeof(float));
    cudaMalloc((void**) &d_out, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    vector_add<<<M, T>>>(d_out, d_a, d_b, N);
    cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("[");
    // for(int i = 0; i < N; i++) {
    //     printf("%2.0f", out[i]);
    // }
    // printf("]");

    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("PASSED");
    
    cudaFree(d_out);
    cudaFree(d_a);
    cudaFree(d_b);
    
    free(out);
    free(a);
    free(b);

    return 0;
}