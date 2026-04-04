# 9.7.14.6.2.4-5. Matrix Fragments for sparse mma.m16n8k8/k32 with .tf32 and Integer Types

Covers fragment distributions for sparse `mma.m16n8k8` (tf32) and `mma.m16n8k32` (u8/s8) operations. Both use 2:4 structured sparsity on matrix A. Includes per-thread fragment layouts for compressed A, full B, accumulator C/D, and metadata.
