# 2.2 Writing CUDA SIMT Kernels — Memory Performance, Atomics, and Occupancy

Covers achieving coalesced global memory access (32-byte transaction alignment), shared memory bank conflicts and padding fix, two matrix-transpose kernel examples (naive global vs. shared memory staging), atomic operations via `cuda::atomic_ref`, cooperative groups for cross-block synchronization, and SM occupancy calculation based on thread count, shared memory, and register limits.
