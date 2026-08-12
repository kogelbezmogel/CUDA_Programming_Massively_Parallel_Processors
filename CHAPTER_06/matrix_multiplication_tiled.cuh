#ifndef __MATRIX_MULTIPLICATION_TILED__
#define __MATRIX_MULTIPLICATION_TILED__

void __global__ matrix_multiplication_tiled(float *A, float *B, float *C, int N, int K, int M);


#endif /* __MATRIX_MULTIPLICATION_TILED__ */