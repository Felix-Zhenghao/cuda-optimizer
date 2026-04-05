# 3.2–3.3. Independent Thread Scheduling & On-chip Shared Memory

Volta and later GPUs support Independent Thread Scheduling, giving each thread its own program counter and call stack for sub-warp divergence. On-chip memory per SM includes per-thread registers, shared memory, read-only constant cache, and read-only texture cache.
