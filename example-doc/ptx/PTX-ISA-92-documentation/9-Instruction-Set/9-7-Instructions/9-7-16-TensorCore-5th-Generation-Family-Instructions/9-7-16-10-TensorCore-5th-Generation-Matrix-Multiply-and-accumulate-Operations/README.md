# 9.7.16.10. TensorCore 5th Generation Matrix Multiply-and-Accumulate Operations

- **9.7.16.10.1-3. Transpose and Negate** -- Transpose and negate operations for MMA input matrices A and B.
- **9.7.16.10.4. Packing Formats** -- Element packing in Tensor Memory and shared memory.
- **9.7.16.10.5. Data Path Layout Organization** -- Layout mappings (A-F) from matrix elements to Tensor Memory lanes.
- **9.7.16.10.6. Shared Memory Layout and Swizzling** -- K-major vs non-K-major layout and swizzling modes for matrix B.
- **9.7.16.10.7. Block Scaling for tcgen05.mma** -- Scale factor layouts for block-scaled MMA operations.
- **9.7.16.10.8. Sparse Matrices** -- Structured sparsity, metadata, and alignment for sparse tcgen05 MMA.
- **9.7.16.10.9. MMA Instructions** -- tcgen05.mma, tcgen05.mma.sp, tcgen05.mma.ws, and tcgen05.mma.ws.sp.
