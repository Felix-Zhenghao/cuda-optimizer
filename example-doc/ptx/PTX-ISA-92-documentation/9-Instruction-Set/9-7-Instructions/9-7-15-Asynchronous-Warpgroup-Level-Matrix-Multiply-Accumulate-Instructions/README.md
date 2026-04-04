# 9.7.15. Asynchronous Warpgroup-Level Matrix Multiply-Accumulate Instructions

- **9.7.15.1. Warpgroup** -- Definition of a warpgroup as four contiguous warps with rank requirements.
- **9.7.15.2. Matrix Shape** -- Supported MxNxK shapes for wgmma.mma_async and wgmma.mma_async.sp.
- **9.7.15.3-4. Matrix Data-types and Block Scaling** -- Data types and block scaling with .kind qualifiers for warpgroup MMA.
- **9.7.15.5. Async Warpgroup MMA using wgmma.mma_async** -- Fragments, shared memory layouts, descriptors, and the wgmma.mma_async instruction.
- **9.7.15.6. Async Warpgroup MMA using wgmma.mma_async.sp** -- Sparse warpgroup MMA: storage, fragments, and the wgmma.mma_async.sp instruction.
- **9.7.15.7. Asynchronous wgmma Proxy Operations** -- Fence, commit, and wait instructions for wgmma synchronization.
