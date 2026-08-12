#ifndef __UTILS__
#define __UTILS__


namespace uti {

    class IntGenerator;

    void fill_tensor_with_gaussian_dist(float *A, int r);
    void fill_memory_with_random_data(float *A, int len, IntGenerator gen);

    
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

} // namespace uti
#endif //__UTILS