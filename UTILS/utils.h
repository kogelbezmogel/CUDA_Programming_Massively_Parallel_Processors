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
                _seed = (7759097958782935 * _seed) % 18055400005099021; 
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


    void print_GPU_info() {
        int device = 0;
        int regis_per_block;
        int shmem_per_block;
        int threa_per_block;
        int regis_per_sm;
        int shmem_per_sm;
        int threa_per_sm;
        int warp_size;
        int sms;

        cudaDeviceGetAttribute(&regis_per_block, cudaDevAttrMaxRegistersPerBlock, device);
        cudaDeviceGetAttribute(&shmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);
        cudaDeviceGetAttribute(&threa_per_block, cudaDevAttrMaxThreadsPerBlock, device);
        cudaDeviceGetAttribute(&regis_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, device);
        cudaDeviceGetAttribute(&shmem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
        cudaDeviceGetAttribute(&threa_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);
        cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);

        printf("GPU ------------------------\n");
        printf("Max registers per block:     %6d floats\n", regis_per_block);
        printf("Max registers per sm:        %6d floats\n", regis_per_sm);
        printf("Max shared memory per block: %6d B = %5lu floats\n", shmem_per_block, shmem_per_block / sizeof(float));
        printf("Max shared memory per sm:    %6d B = %5lu floats\n", shmem_per_sm, shmem_per_sm / sizeof(float));
        printf("Max threads per block:       %6d threads\n", threa_per_block);
        printf("Max threads per sm:          %6d threads\n", threa_per_sm);
        printf("Warp size:                   %6d threads\n", warp_size);
        printf("Amount of SMs:               %6d\n\n", sms);
    }


    template <typename F>
    float elapsed_time(F&& kernel) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float mseconds;
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cudaDeviceSynchronize();
        kernel();
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&mseconds, start, stop);
        return mseconds;
    } 

} // namespace uti
#endif //__UTILS