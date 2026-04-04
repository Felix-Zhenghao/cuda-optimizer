# 9.7.14.5.6. Matrix Fragments for mma.m16n8k4

Describes fragment distribution for `mma.m16n8k4` MMA operation with f64 floating-point type. Elements of the 16x8x4 matrix operation are distributed across warp threads, with each thread holding specific f64 register fragments for matrices A, B, C, and D.
