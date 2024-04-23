#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    int M = 100;
    int N = 50;
    int K = 50;

    int size_A = M * N * sizeof(float);
    int size_B = N * K * sizeof(float);
    int size_C = M * K * sizeof(float);

    float *host_A = (float *)malloc(size_A);
    float *host_B = (float *)malloc(size_B);
    float *host_C = (float *)malloc(size_C);

    for (int i = 0; i < M * N; ++i) host_A[i] = 1.0f;
    for (int i = 0; i < N * K; ++i) host_B[i] = 2.0f;

    float *device_A, *device_B, *device_C;
    cudaMalloc((void **)&device_A, size_A);
    cudaMalloc((void **)&device_B, size_B);
    cudaMalloc((void **)&device_C, size_C);

    cudaMemcpy(device_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

   
    matrixMulBasic<<<gridDim, blockDim>>>(device_A, device_B, device_C, M, N, K);

    cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(host_A);
    free(host_B);
    free(host_C);

    return 0;
}
