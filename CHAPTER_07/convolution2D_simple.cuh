#ifndef __CONV_SIMPLE__
#define __CONV_SIMPLE__

void __global__ simple_convolution2D(float *A, float *F, float *B, int N, int M, int r);

#endif //__CONV_SIMPLE__