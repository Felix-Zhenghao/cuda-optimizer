# 9.7.14.5.3. Matrix Fragments for mma.m8n8k16

Describes fragment distribution for `mma.m8n8k16` MMA operation. Elements of A, B, C, D matrices are distributed across warp threads. Covers integer types (u8, s8) with s32 accumulator. Each thread holds specific b32 register fragments containing packed elements.
