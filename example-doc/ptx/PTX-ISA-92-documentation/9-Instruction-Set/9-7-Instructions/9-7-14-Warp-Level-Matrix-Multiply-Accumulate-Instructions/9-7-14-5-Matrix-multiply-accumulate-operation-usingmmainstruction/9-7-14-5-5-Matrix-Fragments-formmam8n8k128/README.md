# 9.7.14.5.5. Matrix Fragments for mma.m8n8k128

Describes fragment distribution for `mma.m8n8k128` MMA operation with single-bit (b1) type. Each thread holds one b32 register for A and one for B (containing 32 packed b1 elements each). Accumulator C/D uses s32 type with two registers per thread.
