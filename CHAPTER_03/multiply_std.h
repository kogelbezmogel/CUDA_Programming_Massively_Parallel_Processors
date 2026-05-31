#ifndef __MLT__
#define __MLT__


void multiply_matrices_cpu(float *A, float *B, float *C, int N, int K, int M) {
    // A -> N x K
    // B -> K x M
    // C -> N x M
    // A * B = C
    
    float s_value;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
            
            s_value = 0;
            for(int k = 0; k < K; ++k) {
                s_value += A[i*K + k] * B[k*M + j];
            }
            C[i * M + j] = s_value;
        }
    }
}


void multiply_vector_matrix_cpu(float *A, float *B, float *C, int N, int M) {
    // A -> 1 x N
    // B -> N x M
    // C -> 1 x M
    // A * B = C
    
    float s_value;
    for(int j = 0; j < M; ++j) {
            
        s_value = 0;
        for(int i = 0; i < N; ++i) {
            s_value += A[i] * B[i * M + j];
        }
    
        C[j] = s_value;
    }
    
}




#endif