#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrixMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < N)
            s_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0;

        if (col < K && t * TILE_SIZE + threadIdx.y < N)
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            s_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K)
        C[row * K + col] = sum;
}

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

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    matrixMulTiled<<<gridDim, blockDim>>>(device_A, device_B, device_C, M, N, K);

    cudaMemcpy(host_C, device_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(host_A);
    free(host_B);
    free(host_C);

    return 0;
}
