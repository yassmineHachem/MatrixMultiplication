%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <openacc.h>

#define TILE_SIZE 16

void multiply(float *A, float *B, float *C, int M, int N, int K) {
    #pragma acc parallel loop collapse(2) present(A[0:M*N], B[0:K*N], C[0:M*K]) independent
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < K; j += TILE_SIZE) {
            for (int ii = i; ii < i + TILE_SIZE && ii < M; ii++) {
                for (int jj = j; jj < j + TILE_SIZE && jj < K; jj++) {
                    float sum = 0;
                    for (int k = 0; k < N; k++) {
                        sum += A[ii * N + k] * B[jj * N + k];
                    }
                    C[ii * K + jj] = sum;
                }
            }
        }
    }
}

double get_elapsed_time(struct timeval start, struct timeval stop) {
    return (double)(stop.tv_sec - start.tv_sec) * 1000.0 +
           (double)(stop.tv_usec - start.tv_usec) / 1000.0;
}

int main() {
    int M = 100;
    int N = 50;
    int K = 50;
    float *A, *B, *C;

    A = (float*) malloc(M * N * sizeof(float));
    B = (float*) malloc(K * N * sizeof(float));
    C = (float*) malloc(M * K * sizeof(float));

    for (int i = 0; i < M * N; i++) {
        A[i] = rand() % 10;
    }
    for (int i = 0; i < N * K; i++) {
        B[i] = rand() % 10;
    }

    float *B_transposed = (float*) malloc(N * K * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B_transposed[j * N + i] = B[i * K + j];
        }
    }

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    multiply(A, B_transposed, C, M, N, K);

    gettimeofday(&stop, NULL);
    double elapsed_time = get_elapsed_time(start, stop);

    printf("Execution time for tiled openacc with transposed matrix: %.2f ms\n", elapsed_time);

    free(A);
    free(B);
    free(B_transposed);
    free(C);

    return 0;
}
