void stencil_sequential(float *A, float *W, int N, int K, int M, int rank, int sweep_count);
void fill_tensor_with_stencil_data(float *A, int N, int K, int M);
void fill_stencil_weights(float *W, int rank, float h=1.0);


void stencil_sequential(float *A, float *W, int N, int K, int M, int rank, int sweep_count) {
    
    float *A_1 = A;
    float *A_2 = new float[N * K * M];
    float *swap;

    // coping constraints
    for(int pla = 0; pla < M; ++pla) {
        for(int col = 0; col < N; ++col) {
            A_2[pla * N*K + col*N + 0] = A_1[pla * N*K + col*N + 0];
            A_2[pla * N*K + col*N + N-1] = A_1[pla * N*K + col*N + N-1];
        }
    }
    for(int pla = 0; pla < M; ++pla) {
        for(int row = 0; row < N; ++row) {
            A_2[pla * N*K + (K-1)*N + row] = A_1[pla * N*K + (K-1)*N + row];
            A_2[pla * N*K + 0*N + row] = A_1[pla * N*K + 0*N + row];
        }
    }
    for(int col = 0; col < K; ++col) {
        for(int row = 0; row < N; ++row) {
            A_2[(M-1) * N*K + col*N + row] = A_1[(M-1) * N*K + (K-1)*N + row];
            A_2[0 * N*K + (K-1)*N + row] = A_1[0 * N*K + (K-1)*N + row];
        }
    }

    // sweeping
    float temp;
    for(int sweep = 0; sweep < sweep_count; ++sweep) {

        for(int pla = 1; pla < M-1; ++pla) {
            for(int col = 1; col < K-1; ++col) {
                for(int row = 1; row < N-1; ++row) {
                    
                    temp = 0;
                    for(int i = -rank; i < rank+1; ++i) {
                        temp += A_1[pla * N*K + col * N + row + i] * W[i + rank];
                    }
                    for(int j = -rank; j < rank+1; ++j) {
                        temp += A_1[pla * N*K + (col+j) * N + row] * W[j + rank];
                    }
                    for(int k = -rank; k < rank+1; ++k) {
                        temp += A_1[(pla+k) * N*K + col * N + row] * W[k + rank];
                    }

                    A_2[pla * N*K + col * N + row] = temp;
                }
            }
        }
        //swaping arrays
        swap = A_1;
        A_1 = A_2;
        A_2 = swap;
        // at the end of the loop A_1 has the newest values and A_2 will be filled and swaped in the next iteration
    }

    // coping the array if necessery
    if(A == A_2) {
        for(int pla = rank; pla < M-rank; ++pla) {
            for(int col = rank; col < K-rank; ++col) {
                for(int row = rank; row < N-rank; ++row) {
                    A[pla*N*K + col*N + row] = A_1[pla*N*K + col*N + row];
                }
            }
        }
        // now the newest values are at the same address as input array A
    }
}


// 
void fill_tensor_with_stencil_data(float *A, int N, int K, int M) {

}


// it sets default weights for stencil
void fill_stencil_weights(float *W, int rank, float h) {
    W[0] = -1;
    W[1] = 0;
    W[2] = 1;
    float coeff = 1 / (2*h);
    
    for (int step = 0; step < rank; ++step) {
                
        for(int i = step; i < 2*step+1; ++i) {
            W[i] = W[i] - W[i-step]; 
        }
        coeff *= 1 / (2*h);
    }

    for(int i = 0; i < 2*rank+1; ++i) {
        W[i] *= coeff;
    }
}