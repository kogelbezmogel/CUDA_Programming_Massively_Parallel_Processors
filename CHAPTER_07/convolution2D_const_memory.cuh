#ifndef __CONV_CONST_MEM__
#define __CONV_CONST_MEM__


void __global__ simple_convolution2D_const_memory(float *A, float *B, int N, int M, int r);

#endif // __CONV_CONST_MEM__