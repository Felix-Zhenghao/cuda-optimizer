Documents the sparse `wgmma.mma_async.sp` instruction and its structured sparsity requirements.

- **9-7-15-6-1-Sparse-matrix-storage**: Structured sparsity storage format for matrix A: 2:4 granularity for f16/bf16/FP8/int8, 1:2 for tf32; metadata operand encoding
- **9-7-15-6-2-Matrix-fragments**: Per-thread register fragments for sparse matrix A across m64nNk16, m64nNk32, and m64nNk64 shapes
- **9-7-15-6-3-wgmma.mma_async.sp**: Full sparse `wgmma.mma_async.sp` instruction reference with sparsity metadata and selector operands; requires PTX ISA 8.2 and `sm_90a`
