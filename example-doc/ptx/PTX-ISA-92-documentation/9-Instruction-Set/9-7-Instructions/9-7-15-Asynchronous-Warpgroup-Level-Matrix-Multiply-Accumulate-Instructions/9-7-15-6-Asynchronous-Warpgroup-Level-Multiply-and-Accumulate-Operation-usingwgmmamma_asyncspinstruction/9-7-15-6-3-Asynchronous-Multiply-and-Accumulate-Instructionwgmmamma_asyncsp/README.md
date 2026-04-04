# 9.7.15.6.3. Asynchronous Multiply-and-Accumulate Instruction: wgmma.mma_async.sp

Documents the `wgmma.mma_async.sp` instruction for sparse matrix multiply-accumulate across a warpgroup. Supports f16, bf16, tf32, integer, and alternate FP types. Sparse matrix A in registers with metadata; dense B from shared memory. Requires fence/commit/wait pattern.
