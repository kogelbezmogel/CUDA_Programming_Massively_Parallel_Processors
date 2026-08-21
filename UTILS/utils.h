#ifndef __UTILS__
#define __UTILS__


#include <cmath>
#include <cassert>
#include <stdio.h>


namespace uti {

    class IntGenerator {
        public:
            IntGenerator(int a, int b, int seed = 967): _seed(seed), _a(a), _b(b) { };

            int next() { 
                _seed = (1091 * _seed + 1093) % 101; 
                return _seed % (_b - _a) + _a;
            }

        private:
            int _seed;
            int _a;
            int _b;
    };


    void fill_tensor_with_gaussian_dist(float *A, int r);
    void fill_memory_with_random_data(float *A, int len, IntGenerator gen);
    void check_abs_error(float *A, float *B, int len, float max_error = 1e-4);

    
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


    void fill_memory_with_random_data(float *A, int len, IntGenerator gen) {
        for(int i = 0; i < len; ++i) {
            A[i] = gen.next();
        }
    }

    void check_abs_error(float *A, float *B, int len, float max_error) {
        float err = 0.0f;
        
        for(int i = 0; i < len; ++i) {
            // printf("%4.0f <?> %4.0f\n", A[i], B[i]);
            err += abs(A[i] - B[i]);
        }
        // printf("error %10.1f\n", err);
        assert(err <= max_error);
    }

} // namespace uti
#endif //__UTILS