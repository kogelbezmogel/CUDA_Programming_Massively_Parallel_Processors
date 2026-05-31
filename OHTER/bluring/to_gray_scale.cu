#include <cuda_runtime.h>
#include <math.h>
#include "to_gray_scale.h"

__global__ void convert_to_gray_kenrel(float* gray_img_ptr_device,  float* rgb_img_ptr_device, int gray_img_len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int r, g, b;

    if(id < gray_img_len) {
        r = rgb_img_ptr_device[3*id];
        g = rgb_img_ptr_device[3*id + 1];
        b = rgb_img_ptr_device[3*id + 2];
        gray_img_ptr_device[id] = 0.30*r + 0.59*g + 0.11*b;
    }
}


void convert_to_gray(float* gray_img_ptr,  float* rgb_img_ptr, int rgb_img_len) {
    int gray_img_len = (int) rgb_img_len / 3;
    int block_size = 1024;
    int block_num = (int) ceil( (float) gray_img_len / block_size);
    float *gray_img_ptr_device, *rgb_img_ptr_device;
    
    cudaMalloc((void**) &gray_img_ptr_device, gray_img_len * sizeof(float));
    cudaMalloc((void**) &rgb_img_ptr_device, rgb_img_len * sizeof(float));

    cudaMemcpy(rgb_img_ptr_device, rgb_img_ptr, rgb_img_len * sizeof(float), cudaMemcpyHostToDevice);
    convert_to_gray_kenrel<<<block_num, 1024>>>(gray_img_ptr_device, rgb_img_ptr_device, gray_img_len);
    cudaMemcpy(gray_img_ptr, gray_img_ptr_device, gray_img_len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gray_img_ptr_device);
    cudaFree(rgb_img_ptr_device);
}
