import torch
import triton
import triton.language as tl

@triton.jit
def vec_add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    z_ptr,  # *Pointer* to output vector.
    n_elements,      # *Size* of the vectors.
    BLOCK_SIZE: tl.constexpr  # Number of elements each program should process.
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(0)  # We use a 1D launch grid so axis is 0.

    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements

    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size. Then do the addition and store the result to z.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y

    # Write x + y back to DRAM. Mask out any out-of-bounds elements.
    tl.store(z_ptr + offsets, z, mask=mask)