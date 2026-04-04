# 9.7.15.5.2. Asynchronous Multiply-and-Accumulate Instruction: wgmma.mma_async

Documents the `wgmma.mma_async` instruction performing D = A * B + D across a warpgroup asynchronously. Supports f16, bf16, tf32, integer, sub-byte, single-bit, and alternate FP types with optional block scaling. Matrix B sourced from shared memory via descriptor. Requires fence/commit/wait synchronization.
