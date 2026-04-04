# 9.7.15.6.1. Sparse Matrix Storage

Defines structured sparsity for warpgroup-level `wgmma.mma_async.sp` operations. Sparsity granularity depends on data type: 2:4 for f16/bf16/tf32/u8/s8 and 4:8 for u4/s4 sub-byte types. Metadata encodes non-zero element positions using 2-bit selectors per sub-chunk.
