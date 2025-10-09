# Flash-Attention-with-Triton

A comprehensive implementation and exploration of FlashAttention using Triton, with comparisons to traditional CUDA approaches. This project demonstrates how to implement memory-efficient attention mechanisms that scale to long sequences while maintaining numerical accuracy.

## 🚀 What This Project Contains

This repository provides:
- **FlashAttention implementation in Triton** - A Python-like approach to GPU kernel programming
- **Traditional CUDA examples** - For understanding low-level GPU programming concepts
- **Detailed explanations** - Step-by-step breakdowns of concepts, algorithms, and implementations
- **Performance comparisons** - Memory usage and speed analysis between different approaches

## 📚 Background: Why FlashAttention Matters

### The Problem with Standard Attention

Standard scaled dot-product attention computes:

```
Attention(Q,K,V) = softmax(QK^T / √d) V
```

For sequences of length `n`, this requires:
- **Memory**: O(n²) to store the full attention matrix
- **DRAM Traffic**: Repeatedly reading/writing the large attention matrix
- **Scalability**: Becomes prohibitive for long sequences (>2K tokens)

### FlashAttention's Solution

FlashAttention solves this by:
1. **Tiling**: Breaking computation into blocks that fit in fast GPU memory (SRAM)
2. **Fusing**: Combining QK^T → softmax → (softmax·V) into a single kernel
3. **Streaming**: Never materializing the full n×n attention matrix
4. **Stability**: Using numerically stable softmax with on-the-fly corrections

**Result**: Same mathematical output, O(n·d) memory usage, much faster execution.

## 🏗️ Project Structure

```
Flash-Attention-with-Triton/
├── README.md                           # This file
├── FlashAttentionTriton.py            # Main Triton implementation
├── FlashAttention/
│   ├── FlashAttention.md              # Detailed mathematical explanation
│   └── FlashAttentionBlocked.py       # Blocked implementation example
├── Triton/
│   ├── Triton.md                      # Triton programming guide
│   └── vec_add.py                     # Simple Triton example
└── CUDA/
    ├── CUDA.md                        # CUDA basics and concepts
    ├── vec_add.cu                     # CUDA vector addition
    └── matrix_AddMul.cu               # CUDA matrix operations
```

## 🔧 Key Features

### FlashAttention Implementation
- **Memory Efficient**: O(n·d) instead of O(n²) memory usage
- **Numerically Stable**: Streaming softmax with max correction
- **Fused Operations**: Single kernel for entire attention computation
- **Configurable Block Sizes**: Tunable for different hardware

### Triton Advantages
- **Python-like Syntax**: No more cryptic CUDA code
- **Automatic Optimization**: Memory coalescing and shared memory handled automatically
- **Rapid Prototyping**: Write and test kernels in minutes
- **Easy Debugging**: Clear error messages and device printing

### Educational Value
- **Step-by-step Explanations**: From basic concepts to advanced implementations
- **Comparative Analysis**: CUDA vs Triton approaches
- **Real Examples**: Working code with detailed comments
- **Performance Insights**: Understanding GPU memory hierarchies

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch triton numpy
```

### Running the Examples

1. **Basic Triton Vector Addition**:
```bash
python Triton/vec_add.py
```

2. **FlashAttention Implementation**:
```bash
python FlashAttentionTriton.py
```

3. **CUDA Examples** (requires CUDA toolkit):
```bash
cd CUDA/
nvcc -o vec_add vec_add.cu
./vec_add
```

## 📖 Learning Path

### 1. Start with GPU Basics
- Read [`CUDA/CUDA.md`](CUDA/CUDA.md) for fundamental GPU programming concepts
- Understand threads, blocks, grids, and memory management
- Run [`CUDA/vec_add.cu`](CUDA/vec_add.cu) to see basic CUDA in action

### 2. Learn Triton
- Study [`Triton/Triton.md`](Triton/Triton.md) for Triton-specific concepts
- Understand the `@triton.jit` decorator and `tl.` operations
- Practice with [`Triton/vec_add.py`](Triton/vec_add.py)

### 3. Understand FlashAttention
- Read [`FlashAttention/FlashAttention.md`](FlashAttention/FlashAttention.md) for the mathematical foundation
- Learn about streaming softmax and numerical stability
- Study the blocked implementation approach

### 4. Implement and Experiment
- Examine [`FlashAttentionTriton.py`](FlashAttentionTriton.py) for the complete implementation
- Modify block sizes and see performance impacts
- Compare with standard attention implementations

## 🧮 Key Algorithms Explained

### Streaming Softmax
The core innovation enabling memory efficiency:

```python
# Traditional: compute all at once
scores = Q @ K.T / sqrt(d)
attention = softmax(scores) @ V

# FlashAttention: process in blocks
for each key_block:
    partial_scores = Q_block @ K_block.T / sqrt(d)
    # Update running max and denominator
    new_max = max(old_max, max(partial_scores))
    correction_factor = exp(old_max - new_max)
    # Apply corrections and accumulate
    running_sum = running_sum * correction_factor + sum(exp(partial_scores - new_max))
    output = output * correction_factor + exp(partial_scores - new_max) @ V_block
```

### Grid Calculation Pattern
Understanding how to launch the right number of GPU programs:

```python
# Always use ceiling division to cover all elements
grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

# For 2D operations (matrices)
grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
```

## 📊 Performance Characteristics

### Memory Usage Comparison
| Implementation | Memory Complexity | Typical Usage (n=4096, d=64) |
|----------------|-------------------|-------------------------------|
| Standard Attention | O(n²) | ~67MB for attention matrix |
| FlashAttention | O(n·d) | ~1MB for Q,K,V storage |

### Speed Benefits
- **Short sequences (n<1024)**: Comparable performance
- **Medium sequences (1024<n<4096)**: 2-3x speedup
- **Long sequences (n>4096)**: 5-10x speedup + enables previously impossible sizes

## 🔍 Code Highlights

### Triton Kernel Structure
```python
@triton.jit
def flash_attention_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, ...):
    # Get program IDs for 2D tiling
    pid_m = tl.program_id(0)  # Query blocks
    pid_n = tl.program_id(1)  # Key/Value blocks
    
    # Calculate memory offsets
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load data with proper masking
    Q_block = tl.load(Q_ptr + offsets, mask=mask)
    
    # Compute attention with streaming softmax
    scores = tl.dot(Q_block, K_block) / math.sqrt(d)
    # ... streaming softmax logic ...
    
    # Store results
    tl.store(O_ptr + offsets, output, mask=mask)
```

### Key Triton Concepts
- **`tl.program_id()`**: Which "program" (thread block) am I?
- **`tl.arange()`**: Create offset vectors for memory access
- **`[:, None]` and `[None, :]`**: Broadcasting for matrix operations
- **Masking**: Always mask memory operations for safety

## 🛠️ Customization and Tuning

### Block Size Selection
```python
# Typical good choices (powers of 2)
BLOCK_SIZE = 64   # Good starting point
BLOCK_SIZE = 128  # Often optimal for many GPUs
BLOCK_SIZE = 256  # For very large sequences
```

### Memory vs Compute Trade-offs
- **Larger blocks**: Better compute utilization, more memory usage
- **Smaller blocks**: Less memory, more kernel launches
- **Optimal size**: Depends on sequence length and GPU architecture

## 🎯 Use Cases

### When to Use FlashAttention
- **Long sequences**: >1024 tokens where memory becomes limiting
- **Batch processing**: Multiple sequences simultaneously
- **Memory-constrained environments**: Limited GPU VRAM
- **Production deployments**: Where efficiency matters

### When Standard Attention is Fine
- **Short sequences**: <512 tokens
- **Prototyping**: When implementation speed matters more than runtime speed
- **Educational purposes**: Understanding basic attention mechanisms

## 🤝 Contributing

This project is designed for learning and experimentation. Feel free to:
- Add new kernel implementations
- Optimize existing code
- Add more detailed explanations
- Create visualization tools
- Benchmark on different hardware

## 📚 Further Reading

### Papers
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

### Documentation
- [Triton Documentation](https://triton-lang.org/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Attention Mechanisms](https://pytorch.org/docs/stable/nn.html#attention-mechanisms)

## 🏆 Key Takeaways

1. **FlashAttention doesn't change the math** - it changes the order of operations for efficiency
2. **Triton makes GPU programming accessible** - Python-like syntax with automatic optimizations
3. **Memory hierarchy matters** - Understanding SRAM vs DRAM is crucial for performance
4. **Blocking is powerful** - Tiling strategies apply beyond just attention mechanisms
5. **Numerical stability is critical** - Streaming algorithms need careful handling of precision

---

*This project demonstrates that with the right tools (Triton) and algorithms (FlashAttention), complex GPU programming becomes accessible while achieving state-of-the-art performance. Happy kernel hacking! 🚀*