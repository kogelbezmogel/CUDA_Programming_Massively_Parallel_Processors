void matrix_multiplication_cpu(float *A, float *B, float *C, int N, int K, int M) {
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