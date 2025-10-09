import torch

def flash_attention_blocked(Q, K, V, block_size=64):
    """
    Blocked Flash Attention
    Q, K, V: [seq_len, dim]
    Returns: O [seq_len, dim]
    """

    L, d = Q.shape
    O = torch.zeros_like(Q)

    # Looping
    for i in range(0, L, block_size):
        Q_block = Q[i : i + block_size]  # [B, d]
        acc_block = torch.zeros_like(Q_block)
        max_block = None

        # Loop over key/value blocks
        for j in range(0, L, block_size):
            K_block = K[i : i + block_size]  # [B, d]
            V_block = V[i : i + block_size]  # [B, d]

            # Step 1: Compute raw scores for this block
            scores = (Q_block @ K_block.T) / (d ** 0.5)  # [B, B]

            # Step 2: Numerically stable softmax (max subtraction)
            if max_block is None:
                max_block = scores.max(dim=-1, keepdim=True).values # [B, 1]
            scores_exp = torch.exp(scores - max_block)           # [B, B]

            # Step 3: Normalize within block
            denom = scores_exp.sum(dim=-1, keepdim=True)         # [B, 1]
            scores_softmax = scores_exp / denom                  # [B, B]

            # Step 4: Multiply with V_block and accumulate
            acc_block += scores_softmax @ V_block               # [B, d]
        
        # Store accumulated output for this query block
        O[i:i+block_size] = acc_block

    return O

seq_len, d = 512, 64
block_size = 64
Q = torch.randn(seq_len, d, device='cuda')
K = torch.randn(seq_len, d, device='cuda')
V = torch.randn(seq_len, d, device='cuda')

O = flash_attention_blocked(Q, K, V, block_size)
print(O.shape)  # [512, 64]

            





