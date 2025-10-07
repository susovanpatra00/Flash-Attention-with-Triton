#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)

inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n",
                error_code,
                cudaGetErrorString(error_code),
                file,
                line);
        fflush(stderr);
        exit(error_code);
    }
}

// ==================== MATRIX ADDITION ====================
__global__ void matrixAdd(const float *A, const float *B, float *C, int rows, int cols)
{
    // Compute thread's row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col; // flatten 2D index to 1D
        C[idx] = A[idx] + B[idx];
    }
}

// ==================== MATRIX MULTIPLICATION ====================
__global__ void matrixMul(const float *A, const float *B, float *C, int M, int N, int K)
{
    // Compute thread's row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main()
{
    // Matrix sizes
    int rowsA = 4, colsA = 3;
    int rowsB = 3, colsB = 4; // for multiplication
    int rowsC = rowsA, colsC = colsB; // result of A*B

    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = rowsB * colsB * sizeof(float);
    size_t sizeAdd = rowsA * colsA * sizeof(float);
    size_t sizeMul = rowsC * colsC * sizeof(float);

    // Host memory allocation
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_Cadd = (float *)malloc(sizeAdd);
    float *h_Cmul = (float *)malloc(sizeMul);

    // Initialize matrices
    for (int i = 0; i < rowsA * colsA; i++) h_A[i] = i + 1;
    for (int i = 0; i < rowsB * colsB; i++) h_B[i] = i + 1;

    // Device memory allocation
    float *d_A, *d_B, *d_Cadd, *d_Cmul;
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void **)&d_Cadd, sizeAdd));
    CUDA_CHECK(cudaMalloc((void **)&d_Cmul, sizeMul));

    // Copy host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // -------- MATRIX ADDITION --------
    /*
    What dim3 is
        dim3 is a struct in CUDA used to store 3D dimensions. It can hold x, y, z integers:
            dim3 blockDim(x, y, z);
        You can think of it as a 3D vector of integers: {x, y, z}.
        Default values: if you only give x, then y = z = 1.

    Why CUDA needs dim3 ??
        CUDA threads are organized as:
            Threads → small execution units
            Blocks → group of threads
            Grid → group of blocks
        Threads inside a block have indices: threadIdx.x, threadIdx.y, threadIdx.z
        Blocks inside a grid have indices: blockIdx.x, blockIdx.y, blockIdx.z
        Each dimension has a size: blockDim.x/y/z, gridDim.x/y/z
        dim3 is used to specify these sizes when launching kernels.
        
    Example in 2D:
        Suppose you have a 16x16 thread block and a 32x32 block grid:
            dim3 threadsPerBlock(16, 16); // each block has 16x16 threads
            dim3 blocksPerGrid(32, 32);   // grid has 32x32 blocks
        Then inside the kernel:
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
        This computes the global row and column indices for each thread.
    */

    // threadsPerBlock = 16x16 threads per block
    dim3 threadsPerBlockAdd(16, 16);
    // blocksPerGrid = enough blocks to cover all rows and columns
    dim3 blocksPerGridAdd(
        (colsA + threadsPerBlockAdd.x - 1) / threadsPerBlockAdd.x,
        (rowsA + threadsPerBlockAdd.y - 1) / threadsPerBlockAdd.y
    );

    // Launch kernel
    matrixAdd<<<blocksPerGridAdd, threadsPerBlockAdd>>>(d_A, d_A, d_Cadd, rowsA, colsA);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_Cadd, d_Cadd, sizeAdd, cudaMemcpyDeviceToHost));

    printf("Matrix Addition (A + A):\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++)
            printf("%f ", h_Cadd[i * colsA + j]);
        printf("\n");
    }

    // -------- MATRIX MULTIPLICATION --------
    dim3 threadsPerBlockMul(16, 16); // 16x16 threads per block
    dim3 blocksPerGridMul(
        (colsB + threadsPerBlockMul.x - 1) / threadsPerBlockMul.x,
        (rowsA + threadsPerBlockMul.y - 1) / threadsPerBlockMul.y
    );

    matrixMul<<<blocksPerGridMul, threadsPerBlockMul>>>(d_A, d_B, d_Cmul, rowsA, colsA, colsB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_Cmul, d_Cmul, sizeMul, cudaMemcpyDeviceToHost));

    printf("\nMatrix Multiplication (A * B):\n");
    for (int i = 0; i < rowsC; i++) {
        for (int j = 0; j < colsC; j++)
            printf("%f ", h_Cmul[i * colsC + j]);
        printf("\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Cadd));
    CUDA_CHECK(cudaFree(d_Cmul));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_Cadd);
    free(h_Cmul);

    return 0;
}
