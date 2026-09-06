#define BLOCK 32
#define MAX_SIZE 3072
#define REPS 3
#define WARM_UP 1

#include "../UTILS/utils.h"
#include "../UTILS/matrix_multiplication.h"
#include "matrix_multiplication.cuh"
#include "matrix_multiplication_tiled.cuh"
#include "matrix_multiplication_coarsed.cuh"

#include <cmath>
#include <fstream>
#include <functional>

void clear_cache() {
    int l2_size;
    float *Dummy;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, 0);
    cudaMalloc((void**) &Dummy, 2 * l2_size * sizeof(float));
    cudaMemset((void *) Dummy, 0, 2 * l2_size * sizeof(float));
}


int main() {

    float time_sum = 0.0;
    float time_avg = 0.0;
    float time = 0.0;

    dim3 grid_size;
    dim3 block_size;

    int N;
    int K;
    int M;

    // calulating CFACTOR
    int cfactor = 1;
    int device = 0;
    int threa_per_sm;
    int sms;

    cudaDeviceGetAttribute(&threa_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);

    // printf("Maximum blocks per sm according to shmem usage:     %d\n", (int) std::floor(shmem_per_sm / (2 * BLOCK * BLOCK * sizeof(float))));
    // printf("Maximum blocks per sm according to registers usage: %d\n", (int) std::floor(regis_per_sm / (BLOCK * BLOCK * 22)));

    float *A_h, *B_h, *C_h, *A_d, *B_d, *C_d, *C_t;
    uti::IntGenerator gen(0, 10);

    std::string file_name = "./data/mul_coarsed_data2.csv";
    std::fstream file(file_name, std::fstream::in | std::fstream::out | std::fstream::app);
    file << "size;";
    for(int i = 0; i < REPS; ++i) { file << "test_" << i << ";"; }
    file << "avg\n";

    for(int test = 64; test <= MAX_SIZE; test += 64) {
        file << test << ";";

        N = test;
        K = N;
        M = N;

        cfactor = std::ceil((float) (1.0 * N * M) / (sms * threa_per_sm));
        cfactor = 16;

        A_h = new float[N * K];
        B_h = new float[K * M];
        C_h = new float[N * M];
        C_t = new float[N * M];

        cudaMalloc((void**) &A_d, N * K * sizeof(float));
        cudaMalloc((void**) &B_d, K * M * sizeof(float));
        cudaMalloc((void**) &C_d, N * M * sizeof(float));

        time_sum = 0.0;
        time_avg = 0.0;
        printf("N = %4d cf = %3d | ", test, cfactor);
        for(int repetition = 0; repetition < REPS + WARM_UP; ++repetition) {

            uti::fill_memory_with_random_data(A_h, N*K, gen);
            uti::fill_memory_with_random_data(B_h, K*M, gen);

            cudaMemcpy(A_d, A_h, N * K * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(B_d, B_h, K * M * sizeof(float), cudaMemcpyHostToDevice);
            
            grid_size  = dim3((M + cfactor*BLOCK-1) / (cfactor*BLOCK), (N+BLOCK-1) / BLOCK, 1);
            block_size = dim3(BLOCK, BLOCK, 1);

            clear_cache();
            cudaMemset(C_d, 0, N * M * sizeof(float));
            time = uti::elapsed_time([&] {matrix_multiplication_coarsed<<<grid_size, block_size>>>(A_d, B_d, C_d, N, K, M, cfactor);});
            cudaMemcpy(C_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

            if(repetition >= WARM_UP) {
                printf("%7.3fms ", time);
                time_sum += time;
            }

            if(repetition == 0) {
                matrix_multiplication_cpu(A_h, B_h, C_t, N, K, M);
                uti::check_abs_error(C_h, C_t, N*M);
            }

            file << time << ";";
        }
        time_avg = time_sum / REPS;
        file << time_avg << "\n";
        printf("| AvgTime: %7.3f ms\n", time_avg);

        delete A_h;
        delete B_h;
        delete C_h;
        delete C_t;
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
    }

    // closing file
    file.close();

    return 0;
}


