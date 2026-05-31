#include <iostream>

#define N 2
#define M 4

int main() {

    int arr1[N][M];
    //       y  x

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
                arr1[i][j] = i*M + j; 
        }
    }
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j) {
               std::cout << arr1[i][j] << " "; 
        }
    }

    std::cout << "\n\n";
    for(int i = 0; i < N * M; ++i) {
        std::cout << *(*arr1 + i) << " "; 
    }

    std::cout << "\n\n";
    int arr2[N][N][N];
    //       z  y  x    it is row-major order

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            for(int k = 0; k < N; ++k) {
                arr2[i][j][k] = i * N*N + j * N + k; 
            }
        }
    }
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            for(int k = 0; k < N; ++k) {
               std::cout << arr2[i][j][k] << " "; 
            }
        }
    }

    std::cout << "\n\n";
    for(int i = 0; i < N * N * N; ++i) {
        std::cout << *(**arr2 + i) << " "; 
    }

    return 0;
}