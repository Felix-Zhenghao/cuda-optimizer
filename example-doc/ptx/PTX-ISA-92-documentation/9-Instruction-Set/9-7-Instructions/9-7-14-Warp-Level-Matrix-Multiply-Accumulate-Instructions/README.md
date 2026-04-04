# 9.7.14. Warp-Level Matrix Multiply-Accumulate Instructions

- **9.7.14.1. Matrix Shape** -- Supported MxNxK shapes for wmma, mma, and mma.sp instructions across all data types.
- **9.7.14.2. Matrix Data-types** -- Data type specifications for MMA operands including floating-point, integer, sub-byte, and alternate formats.
- **9.7.14.3. Block Scaling for mma.sync** -- Block scaling with .kind qualifiers for scaled matrix multiply D = (A*scale_A)*(B*scale_B)+C.
- **9.7.14.4. Matrix multiply-accumulate using wmma instructions** -- WMMA operations: fragments, storage, load/store, and mma instructions at warp level.
- **9.7.14.5. Matrix multiply-accumulate using mma instruction** -- MMA operations: fragment layouts for all shapes/types, mma instruction, ldmatrix, stmatrix, movmatrix.
- **9.7.14.6. Matrix multiply-accumulate using mma.sp with sparse matrix A** -- Sparse MMA: storage format, fragment layouts, and mma.sp/mma.sp::ordered_metadata instructions.
