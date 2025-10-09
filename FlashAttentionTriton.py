import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, 
                           stride_qm: tl.constexpr, 
                           stride_km: tl.constexpr, 
                           stride_vm: tl.constexpr, 
                           stride_om: tl.constexpr,
                           block_size: tl.constexpr
    ):
    """
    Blocked Flash Attention Kernel in Triton
    Q_ptr, K_ptr, V_ptr: pointers to query/key/value tensors [L, d]
    O_ptr: output tensor pointer
    stride_*: row strides
    block_size: tile size
    """
    row_start = tl.program_id(0) * block_size
    L = tl.program_id(1) * block_size

    # Initialize accumulator for the query block
    acc_block = tl.zeros([block_size, block_size], dtype=tl.float32)

    # Initialize max for numerical stability
    max_block = tl.full([block_size, 1], float('-inf'), dtype=tl.float32)

    # Looping over key/value blocks
    for k_start in range(0, L, block_size):
        # Loading Q, K, V blocks from global memory
        Q_block = tl.load(Q_ptr + row_start * stride_qm + tl.arange(0, block_size)[:, None])
        K_block = tl.load(K_ptr + k_start * stride_km + tl.arange(0, block_size)[None, :])
        V_block = tl.load(V_ptr + k_start * stride_vm + tl.arange(0, block_size)[None, :])

        # Step 1: Computing attention scores
        scores = tl.dot(Q_block, K_block) / (block_size ** 0.5)  # [B, B]

        # Step 2: Numerically stable softmax (max subtraction)
        max_scores = tl.max(scores, axis=1, keepdim=True)  # [B, 1]
        max_block = tl.maximum(max_block, max_scores) 

        # Numerically stable softmax
        scores_exp = tl.exp(scores - max_block)  # [B, B]
        denom = tl.sum(scores_exp, axis=1, keepdim=True)  # [B, 1]
        softmax_block = scores_exp / denom

        # Multiply with V_block and accumulate
        acc_block += tl.dot(softmax_block, V_block) # [B, d]

    # Write back the accumulated output
    tl.store(O_ptr + row_start * stride_om + tl.arange(0, block_size)[:, None], acc_block)


# -----------------------------
# Python wrapper to call kernel
# -----------------------------
def flash_attention_triton(Q, K, V, block_size=64):
    """
    Flash Attention using Triton
    Q, K, V: [seq_len, dim]
    Returns: O [seq_len, dim]
    """
    L, d = Q.shape
    O = torch.zeros_like(Q)

    # Define grid size, Grid dimensions: one program per query block, one per row tile
    grid = ( (L + block_size - 1) // block_size, (L + block_size - 1) // block_size )

    # Launch kernel
    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), K.stride(0), V.stride(0), O.stride(0),
        block_size=block_size
    )

    return O


# -----------------------------
# Test the implementation
# -----------------------------
seq_len, d = 512, 64
Q = torch.randn(seq_len, d, device='cuda', dtype=torch.float32)
K = torch.randn(seq_len, d, device='cuda', dtype=torch.float32)
V = torch.randn(seq_len, d, device='cuda', dtype=torch.float32)

O = flash_attention_triton(Q, K, V, block_size=64)
print(O.shape)  # Expected: [512, 64]


    