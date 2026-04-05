# 4.11.1. Asynchronous Data Copies with LDGSTS

Explains the LDGSTS instruction for asynchronous global-to-shared memory copies that bypass the register file. Covers three patterns: batching loads in conditional code to avoid predicated barriers, software prefetching to pipeline memory loads with computation, and producer-consumer warp specialization for overlapping data movement with compute.
