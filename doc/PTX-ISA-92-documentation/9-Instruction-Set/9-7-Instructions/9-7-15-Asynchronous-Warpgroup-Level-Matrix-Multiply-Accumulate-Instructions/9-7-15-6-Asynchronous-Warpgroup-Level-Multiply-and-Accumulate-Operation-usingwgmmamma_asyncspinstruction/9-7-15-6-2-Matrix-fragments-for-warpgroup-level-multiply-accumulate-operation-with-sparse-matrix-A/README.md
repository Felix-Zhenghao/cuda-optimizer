Documents per-thread register fragments for sparse `wgmma.mma_async.sp` matrix A operands.

- **9-7-15-6-2-1-to-9-7-15-6-2-2**: Fragments for m64nNk32 (f16/bf16, 2:4 sparsity) and m64nNk16 (tf32, 1:2 sparsity) with metadata layout
- **9-7-15-6-2-3**: Fragments for m64nNk64 (FP8 e4m3/e5m2 and int8 s8/u8, 2:4 sparsity) with metadata layout for columns 0-31 and 32-63
