# 9.7.14.1. Matrix Shape

Defines supported MxNxK shapes for warp-level matrix multiply-accumulate operations using `wmma`, `mma`, and `mma.sp` instructions. Covers dense and sparse variants across floating-point (f16, bf16, tf32, f64), integer (u8, s8, u4, s4), sub-byte, single-bit, and alternate formats (e4m3, e5m2, e3m2, e2m3, e2m1) with block scaling options. Lists valid shape tuples and required PTX ISA versions.
