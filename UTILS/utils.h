void fill_tensor_with_gaussian_dist(float *A, int r);
void fill_tensor_test(float *A, int r);
void fill_tensor_with_data(float *A, int N, int K, int M, int seed=967);
void fill_tensor_with_test_data(float *A, int N, int K, int M);

void fill_tensor_with_gaussian_dist(float *A, int r) {
    float coeff = 1.0 / (std::sqrt(2 * M_PI));
    for(int i = -r; i < r+1; ++i) {
        for(int j = -r; j < r+1; ++j) {
            for(int k = -r; k < r+1; ++k){
                A[(i+r)*(2*r+1)*(2*r+1) + (j+r)*(2*r+1) + k+r] = coeff * std::exp( (-0.33) * (i*i + j*j + k*k));
            }
        }
    }
}


void fill_tensor_test(float *A, int r) {
    for(int i = 0; i < 2*r+1; ++i) {
        for(int j = 0; j < 2*r+1; ++j) {
            for(int k = 0; k < 2*r+1; ++k){
                A[i*(2*r+1)*(2*r+1) + j*(2*r+1) + k] = i*(2*r+1)*(2*r+1) + j*(2*r+1) + k;
            }
        }
    }
}


void fill_tensor_with_data(float *A, int N, int K, int M, int seed) {
    int num = seed;
    for(int i = 0; i < N*K*M; ++i) {
        num = (1091 * num + 1093) % 101;
        A[i] = num % 10;
    }
}


void fill_tensor_with_test_data(float *A, int N, int K, int M) {
    for(int i = 0; i < N*K*M; ++i) {
        A[i] = i;
    }
}