# 9.7.16.10.8.1-3. Sparse tcgen05.mma.sp with Various Kinds

Describes structured sparsity for tcgen05 sparse MMA operations across different kinds: kind::tf32 (1:2 sparsity), kind::f16 (2:4 sparsity), and kind::f8f6f4 (2:4 sparsity). Non-zero elements of matrix A are stored contiguously. Metadata selects which elements are non-zero.
