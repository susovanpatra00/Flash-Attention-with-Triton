# CUDA Basics and GPU Programming Notes

This is a summary of key concepts and things learned while exploring CUDA programming for GPU computation.

---

## 1. What CUDA Is

CUDA is NVIDIA’s framework for running C/C++ code on the GPU. It allows writing **kernel functions** that execute in parallel across many threads.

---

## 2. Host vs Device

* **Host**: CPU and system memory (RAM)
* **Device**: GPU and its own memory (VRAM)
* Data must be explicitly copied between host and device using CUDA functions.

---

## 3. Kernel Functions

* Declared with `__global__` and executed on the GPU.
* Called from the CPU.
* Each thread executes the same code but works on different data.

**Example of a kernel:**

```c
__global__ void vecAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // unique thread index
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

---

## 4. Threads, Blocks, and Grids

* **Thread**: smallest execution unit
* **Block**: group of threads
* **Grid**: group of blocks
* `dim3` is used to define 1D/2D/3D sizes for threads and blocks:

```c
dim3 threadsPerBlock(16, 16); // 16x16 threads per block
dim3 blocksPerGrid(
    (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
);
```

---

## 5. Thread Indexing

Threads need to know **which data element they should process**. CUDA provides built-in variables:

* `threadIdx` → thread index inside the block
* `blockIdx` → block index inside the grid
* `blockDim` → size of the block

**Example worker/illustrative kernel:**

```c
__global__ void workerExample() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // absolute index
    printf("Thread %d is working\n", idx);
}
```

* Each thread prints its unique `idx`.
* Helps visualize how threads map to data elements.

---

## 6. Common CUDA Functions and Syntax

| Function                | Purpose                              | Syntax Example                                        |
| ----------------------- | ------------------------------------ | ----------------------------------------------------- |
| `cudaMalloc`            | Allocate memory on GPU               | `cudaMalloc((void**)&d_A, size);`                     |
| `cudaMemcpy`            | Copy data between host and device    | `cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);` |
| `cudaFree`              | Free GPU memory                      | `cudaFree(d_A);`                                      |
| `cudaDeviceSynchronize` | Wait for GPU to finish all tasks     | `cudaDeviceSynchronize();`                            |
| `cudaGetLastError`      | Check for errors after kernel launch | `cudaGetLastError();`                                 |

---

## 7. Memory Management

* GPU memory must be explicitly allocated and freed.
* GPU operations are asynchronous; always synchronize before reading results back.
* Always check for errors to catch issues like out-of-bounds access or invalid memory usage.

---

## 8. Vector and Matrix Operations

* **Vector addition**: each thread adds one pair of elements
* **Matrix addition**: threads arranged in 2D for rows and columns
* **Matrix multiplication**: each thread computes one output cell (dot product of a row and a column)

---

## 9. Key Takeaways

* GPUs are designed for **massive parallelism**
* Proper **memory management** and **thread indexing** are essential
* Error checking and synchronization prevent silent failures
* Once indexing and memory management are clear, writing GPU code becomes straightforward

---
