# 9.7.16. TensorCore 5th Generation Family Instructions

- **9.7.16.1-5. Tensor Memory** -- New per-CTA memory type for Blackwell (sm_100+) TensorCore operations.
- **9.7.16.2. Matrix and Data Movement Shape** -- Supported MxNxK shapes and data movement shapes for tcgen05 operations.
- **9.7.16.3. Major-ness Supported by Strides** -- Leading dimension strides and canonical CuTe layouts for shared memory.
- **9.7.16.4. Matrix Descriptors** -- Shared memory descriptor, instruction descriptor, and zero-column mask descriptor formats.
- **9.7.16.6. Memory Consistency Model** -- Async/sync operation categories, ordering rules, and synchronization patterns.
- **9.7.16.7. Tensor Memory Allocation** -- alloc, dealloc, and relinquish_alloc_permit instructions.
- **9.7.16.8. Tensor Memory Load/Store** -- tcgen05.ld, tcgen05.st, and tcgen05.wait instructions.
- **9.7.16.9. Tensor Memory Data Movement** -- tcgen05.cp, tcgen05.shift, and optional decompression.
- **9.7.16.10. Matrix Multiply-Accumulate Operations** -- Transpose/negate, packing, layouts, scaling, sparsity, and MMA instructions.
- **9.7.16.11. Specialized Synchronization** -- tcgen05.fence for cross-proxy memory ordering.
- **9.7.16.12. Async Synchronization** -- tcgen05.commit and tcgen05.wait for mbarrier-tracked completion.
