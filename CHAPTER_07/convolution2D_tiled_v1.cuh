#ifndef __CONV_TILED_V1__
#define __CONV_TILED_V1__


void __global__ convolution2D_tiled_v1(float *A, float *B, int N, int M, int r);

#endif // __CONV_TILED_V1__