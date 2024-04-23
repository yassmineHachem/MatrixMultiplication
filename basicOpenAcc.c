%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <openacc.h>

void multiply(float *A, float *B, float *C, int M, int N) {
    #pragma acc parallel loop copyin(A[0:M*N], B[0:M*N]) copyout(C[0:M*N]) independent
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            float sum = 0;
            for (int k = 0; k < N; k ++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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
    float *A, *B, *C;

    A = (float*) malloc(M * N * sizeof(float));
    B = (float*) malloc(N * N * sizeof(float));
    C = (float*) malloc(M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 10;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = rand() % 10;
        }
    }

    struct timeval start, stop;
    gettimeofday(&start, NULL);

    multiply(A, B, C, M, N);

    gettimeofday(&stop, NULL);
    double elapsed_time = get_elapsed_time(start, stop);

    printf("Execution time for basic openacc: %.2f ms\n", elapsed_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
