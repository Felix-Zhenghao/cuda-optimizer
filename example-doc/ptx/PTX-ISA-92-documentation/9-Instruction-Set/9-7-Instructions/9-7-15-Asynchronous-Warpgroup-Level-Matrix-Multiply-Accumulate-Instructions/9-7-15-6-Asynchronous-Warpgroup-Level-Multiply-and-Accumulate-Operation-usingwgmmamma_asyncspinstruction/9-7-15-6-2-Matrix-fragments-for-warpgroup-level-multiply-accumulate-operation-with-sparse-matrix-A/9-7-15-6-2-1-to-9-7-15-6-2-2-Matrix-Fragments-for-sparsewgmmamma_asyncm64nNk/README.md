# 9.7.15.6.2.1-2. Matrix Fragments for sparse wgmma.mma_async.m64nNk32

Describes per-thread fragment distribution for sparse `wgmma.mma_async.m64nNk32` with f16/bf16 and tf32 types. Matrix A uses 2:4 structured sparsity. Includes fragment layouts for compressed A in registers, metadata, and accumulator D across warpgroup threads.
