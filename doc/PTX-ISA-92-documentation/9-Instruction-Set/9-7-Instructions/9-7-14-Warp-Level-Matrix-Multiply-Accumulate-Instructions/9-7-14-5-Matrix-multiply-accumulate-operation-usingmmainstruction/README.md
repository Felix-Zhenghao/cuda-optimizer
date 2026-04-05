# 9.7.14.5. Matrix Multiply-Accumulate using `mma` Instructions

- **9-7-14-5-1** — Fragment layout for mma.m8n8k4 with f16 type (4 parallel MMA computations per warp).
- **9-7-14-5-2** — Fragment layout for mma.m8n8k4 with f64 type.
- **9-7-14-5-3** — Fragment layout for mma.m8n8k16 with u8/s8 integer type.
- **9-7-14-5-4** — Fragment layout for mma.m8n8k32 with u4/s4 sub-byte integer type.
- **9-7-14-5-5** — Fragment layout for mma.m8n8k128 with b1 single-bit type.
- **9-7-14-5-6** — Fragment layout for mma.m16n8k4 with tf32 and f64 types.
- **9-7-14-5-7** — Fragment layout for mma.m16n8k8 with f16/bf16, tf32, and f64 types.
- **9-7-14-5-8** — Fragment layout for mma.m16n8k16 with floating-point types (f16/bf16, tf32, fp8).
- **9-7-14-5-9** — Fragment layout for mma.m16n8k16 with integer and fp8 types.
- **9-7-14-5-10** — Fragment layout for mma.m16n8k32 with sub-byte and fp8 types.
- **9-7-14-5-11** — Fragment layout for mma.m16n8k64 with u4/s4 and e2m1 types.
- **9-7-14-5-12** — Fragment layout for mma.m16n8k128 with b1 single-bit type.
- **9-7-14-5-13** — Fragment layout for mma.m16n8k256 with b1 single-bit type.
- **9-7-14-5-14-Multiply-and-Accumulate-Instructionmma** — Full `mma.sync` instruction reference for all supported types and shapes.
- **9-7-14-5-15-Warp-level-matrix-load-instructionldmatrix** — `ldmatrix` instruction for loading matrices from shared memory.
- **9-7-14-5-16-Warp-level-matrix-store-instructionstmatrix** — `stmatrix` instruction for storing matrices to shared memory.
- **9-7-14-5-17-Warp-level-matrix-transpose-instructionmovmatrix** — `movmatrix` instruction for transposing an m8n8 matrix in-register across a warp.
