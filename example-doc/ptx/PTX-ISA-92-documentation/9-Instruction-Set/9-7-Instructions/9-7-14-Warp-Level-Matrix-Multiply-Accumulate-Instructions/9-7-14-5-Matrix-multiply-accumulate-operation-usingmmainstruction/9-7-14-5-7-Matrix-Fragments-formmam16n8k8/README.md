# 9.7.14.5.7. Matrix Fragments for mma.m16n8k8

Describes fragment distribution for `mma.m16n8k8` MMA operation. Covers floating-point types f16, bf16, and tf32. Elements of matrices A, B, C, D are distributed across warp threads with each thread holding specific register fragments for the 16x8x8 shape.
