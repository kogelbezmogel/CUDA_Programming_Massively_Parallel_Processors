// Zip function
// Implement a kernel that adds together each position of arrays a and b and stores it in out.
// You have 1 thread per position.

#define N 4
#define T 256
#define MAX_ERR 1e-5

#include <assert.h>


__global__ void zip(float* out, float* a, float *b, int n) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_id < n) {
        out[thread_id] = a[thread_id] + b[thread_id];
    }
}


int main() {

    float *out, *res, *a, *b;
    out = (float*) malloc(N * sizeof(float));
    res = (float*) malloc(N * sizeof(float));
    a   = (float*) malloc(N * sizeof(float));
    b   = (float*) malloc(N * sizeof(float));
    
    // setting arguments and expected result
    for(int i = 0; i < N; i++) {
        res[i] = i + i;
        a[i] = i;
        b[i] = i;
    }

    float *device_out, *device_a, *device_b;
    cudaMalloc((void**) &device_out, N * sizeof(float));
    cudaMalloc((void**) &device_a, N * sizeof(float));
    cudaMalloc((void**) &device_b, N * sizeof(float));
    
    cudaMemcpy((void*) device_a, (void*) a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) device_b, (void*) b, N * sizeof(float), cudaMemcpyHostToDevice);

    int block_num = (int) ceil((double) N / T);
    
    zip<<<block_num, T>>>(device_out, device_a, device_b, N);

    cudaMemcpy((void*) out, (void*) device_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    //checking the results
    for(int i = 0; i < N; i++) {
        assert(fabs(out[i] - res[i]) < MAX_ERR);
    }
    return 0;
}
