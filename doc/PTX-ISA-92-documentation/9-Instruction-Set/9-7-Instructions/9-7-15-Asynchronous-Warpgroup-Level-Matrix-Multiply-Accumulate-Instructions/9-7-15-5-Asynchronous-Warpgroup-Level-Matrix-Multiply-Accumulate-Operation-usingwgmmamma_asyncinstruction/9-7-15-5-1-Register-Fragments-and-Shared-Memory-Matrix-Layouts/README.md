Covers the data layout specifications for `wgmma.mma_async`: how matrix fragments are distributed across threads in registers and how matrices are stored in shared memory.

- **9-7-15-5-1-1-Register-Fragments**: Per-thread register fragment layouts for matrix A and accumulator D across all shape/type combinations (k8, k16, k32, k256)
- **9-7-15-5-1-2-Shared-Memory-Matrix-Layout**: Shared memory layout strides (LBO, SBO), canonical CuTe layouts, swizzling modes, and the 64-bit matrix descriptor format
