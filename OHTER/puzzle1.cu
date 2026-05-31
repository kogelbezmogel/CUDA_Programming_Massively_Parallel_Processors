// Map function
// function adding 

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define N 4
#define CON 10
#define MAX_ERR 1e-4

__global__ void map(float* out, float* a, float con, int n) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x;

    for(int i = thread_id; i < n; i += stride) {
        out[i] = a[i] + con;
    }
}


int main() {

    // allocating memory on host
    float *res, *out, *a;
    res = (float*) malloc(N * sizeof(float));
    out = (float*) malloc(N * sizeof(float));
    a   = (float*) malloc(N * sizeof(float));

    for(int i = 0; i < N; i++) {
        res[i] = i + CON;
        a[i] = i;
    }

    // allocating memory on device
    float *device_out, *device_a;
    cudaMalloc((void**) &device_a, N * sizeof(float));
    cudaMalloc((void**) &device_out, N * sizeof(float));
    
    // copying argument to device memory
    cudaMemcpy((void*) device_a, (void*) a, N * sizeof(float), cudaMemcpyHostToDevice);

    // function execution
    map<<<8, 64>>>(device_out, device_a, CON, N);    
    
    // extracting resulting data
    cudaMemcpy((void*) out, (void*) device_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // checking the results
    for(int i = 0; i < N; i++) {
        assert(fabs(res[i] - out[i]) < MAX_ERR);
    }

    // freeing allocated memory
    free(res);
    free(out);
    free(a);
    cudaFree(device_out);
    cudaFree(device_a);
    return 0;
}