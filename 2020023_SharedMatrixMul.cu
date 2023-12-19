#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <iomanip>

#define N 16 // Size of NxN Matrix
#define BLOCK_SIZE 8
using namespace std;
// Function to print a matrix
void printMatrix(const int *matrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << std::setw(5) << matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Sequential matrix multiplication
void matrix_multiply(int A[N][N], int B[N][N], int C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
/*
__global__ void matrix_multiply_cuda(int *l,int *m, int *n)
{
    int x=blockIdx.x;
    int y=blockIdx.y;
    __shared__ int p[N];

    int i;
    int k=threadIdx.x;

    n[N*y+x]=0;

  __syncthreads();
   p[k]=l[N*y+k]*m[N*k+x];

  

  for(i=0;i<N;i++)
  n[N*y+x]=n[N*y+x]+p[i];
}
*/
__global__ void matrix_multiply_cuda(int *left, int *right, int *res, int dim) {

    int i,j;
    int temp = 0;

    __shared__ int Left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

    // Row i of matrix left
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

        // Column j of matrix left
        j = tileNUM * BLOCK_SIZE + threadIdx.x;
        i = tileNUM * BLOCK_SIZE + threadIdx.y;
        // Load left[i][j] to shared mem

        Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
        // Load right[i][j] to shared mem

        Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
        // Synchronize before computation
        __syncthreads();

        // Accumulate one tile of res from tiles of left and right in shared mem
        for (int k = 0; k < BLOCK_SIZE; k++) {

            temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
        }
        // Synchronize
        __syncthreads();
    }
    // Store accumulated value to res
    res[row * dim + col] = temp;
}

int main() {
    int A[N][N], B[N][N], C[N][N];
    int *dev_A, *dev_B, *dev_C;

    // Initialize matrices A and B randomly
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;  // Adjust range as needed
            B[i][j] = rand() % 10;
        }
    }

    // Sequential matrix multiplication with timing
    clock_t start_cpu = clock();
    matrix_multiply(A, B, C);
    clock_t end_cpu = clock();
    double cpu_time = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    //Print matrices A, B, and C_cpu (result from CPU)
    std::cout << "Matrix A:" << std::endl;
    printMatrix(&A[0][0], N, N);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(&B[0][0], N, N);
    std::cout << "Result from CPU (C_cpu):" << std::endl;
    printMatrix(&C[0][0], N, N);

    // Allocate memory on the GPU
    cudaMalloc((void**)&dev_A, N * N * sizeof(int));
    cudaMalloc((void**)&dev_B, N * N * sizeof(int));
    cudaMalloc((void**)&dev_C, N * N * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 gridSize(N/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 Blocksize(BLOCK_SIZE,BLOCK_SIZE);

    // CUDA matrix multiplication with timing
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    cudaEventRecord(start_gpu);
    matrix_multiply_cuda<<<gridSize, Blocksize>>>(dev_A, dev_B, dev_C,N);
    cudaEventRecord(end_gpu);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, dev_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate GPU execution time
    float gpu_time;
    cudaEventSynchronize(end_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, end_gpu);

    // Print the result from the GPU (C_gpu)
    std::cout << "Result from GPU (C_gpu):" << std::endl;
    printMatrix(&C[0][0], N, N);

    // Print execution times
    printf("\nMatrix Size %d * %d\n",N,N);
    printf("\nExecution Time:\n");
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f milliseconds\n", gpu_time);

    printf("Speed up: %f\n",cpu_time / (gpu_time/1000));

    // Free allocated memory on the GPU
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return 0;
}
