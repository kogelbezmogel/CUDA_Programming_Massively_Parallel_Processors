#include <stdlib.h>

#define N 10000000

void vector_add(float* out, float* a, float* b, int n) {
    for(int j = 0; j < n; j++) {
        out[j] = a[j] + b[j];
    }
}


int main() {
    float *out, *a, *b;

    a =     (float*) malloc(N * sizeof(float));
    b =     (float*) malloc(N * sizeof(float));
    out =   (float*) malloc(N * sizeof(float));
    
    for(int i = 0; i < N; i++) {
        a[i] = (float) ((i*i) % 1001) / (i % 1001);
        b[i] = (float) (i % 1001); 
    }
    
    vector_add(out, a, b, N);
    
    return 0;
}