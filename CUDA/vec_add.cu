#include <stdio.h>
#include <cuda_runtime.h>

/*
 * CUDA_CHECK Macro:
 * This macro wraps CUDA API calls to provide automatic error checking.
 * Why we need it:
 * - CUDA functions return error codes, but many developers ignore them
 * - Silent failures can lead to incorrect results or crashes later
 * - This macro ensures immediate detection and reporting of CUDA errors
 * - It provides file name and line number for easier debugging
 * - The do-while(false) construct ensures it behaves like a single statement
 */
#define CUDA_CHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)

/*
 * Error checking function that gets called by CUDA_CHECK macro
 * Prints detailed error information and exits if an error occurred
 */
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

/*
 * __global__ Keyword:
 * This is a CUDA function qualifier that indicates this function is a "kernel"
 * Why we need __global__:
 * - Marks the function as executable on the GPU (device) but callable from CPU (host)
 * - Without it, the function would be a regular CPU function
 * - Enables parallel execution across thousands of GPU threads
 * - Each thread executes this function simultaneously with different data
 * - The function must return void and cannot be called recursively
 */
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    // Calculate unique thread index: each thread gets a different 'i' value
    // blockDim.x = threads per block, blockIdx.x = current block index
    // threadIdx.x = thread index within current block
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Boundary check: ensure we don't access memory beyond array bounds
    if (i < N) {
        C[i] = A[i] + B[i];  // Each thread adds one pair of elements
    }
}

int main(){
    int N = 1000;
    size_t size = N * sizeof(float);

    // 2. Allocate memory on CPU (host)
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 3. Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2.0f;
    }

    // 4. Allocate memory on GPU (device)
    float *d_A, *d_B, *d_C;
    /*
     * cudaMalloc Syntax: cudaMalloc(void **devPtr, size_t size)
     * - devPtr: pointer to pointer that will hold the device memory address
     * - size: number of bytes to allocate
     * - Returns: cudaError_t (success/failure status)
     * - Similar to malloc() but allocates on GPU's global memory
     * - GPU memory is separate from CPU memory, requires explicit allocation
     * - (void **) cast is needed because we're passing address of pointer
     */
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // 5. Copy data from host → device
    /*
     * cudaMemcpy Syntax: cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
     * - dst: destination memory address
     * - src: source memory address
     * - count: number of bytes to copy
     * - kind: direction of copy (HostToDevice, DeviceToHost, DeviceToDevice, HostToHost)
     *
     * Why we need this:
     * - CPU and GPU have separate memory spaces
     * - Data must be explicitly transferred between them
     * - cudaMemcpyHostToDevice: copies from CPU RAM to GPU memory
     * - This is a synchronous operation (blocks until copy completes)
     */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 6. Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 7. Copy results back to host
    /*
     * cudaMemcpyDeviceToHost: copies computed results from GPU back to CPU
     * - h_C: destination array on CPU (host)
     * - d_C: source array on GPU (device) containing results
     * - size: number of bytes to copy
     * - cudaMemcpyDeviceToHost: specifies GPU → CPU transfer direction
     */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 8. Verify results
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    // 9. Free memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}