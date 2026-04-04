# 9.7.15.7.1. Asynchronous Multiply-and-Accumulate Instruction: wgmma.fence

Documents `wgmma.fence` which enforces ordering of register accesses between `wgmma.mma_async` and other operations. Must be executed before first wgmma and after last wgmma when registers are used. All threads in the warpgroup must execute it.
