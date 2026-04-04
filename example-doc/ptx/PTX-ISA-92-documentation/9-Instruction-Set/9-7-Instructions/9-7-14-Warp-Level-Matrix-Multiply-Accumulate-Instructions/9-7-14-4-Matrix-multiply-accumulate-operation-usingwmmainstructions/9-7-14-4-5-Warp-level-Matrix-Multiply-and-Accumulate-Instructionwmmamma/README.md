# 9.7.14.4.5. Warp-level Matrix Multiply-and-Accumulate Instruction: wmma.mma

Documents the `wmma.mma` instruction performing D = A * B + C across a warp. Supports f16, bf16, tf32, integer, sub-byte, and single-bit types. Includes satfinite mode for f16, .and/.xor/.popc operations for b1, and optional accumulator type conversion. Requires all threads to execute identically.
