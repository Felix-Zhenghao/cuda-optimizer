# 9.7.14.6.1. Sparse Matrix Storage

Defines structured sparsity for matrix A in `mma.sp` operations. Sparsity granularity is the ratio of non-zero elements in a sub-chunk. Supports 2:4 and 4:8 sparsity patterns depending on data type. Metadata encodes which elements are non-zero using 2-bit selectors per sub-chunk.
