PTX half-precision floating-point instructions for f16, bf16, and their packed SIMD variants (f16x2, bf16x2).

- **9-7-4-1-add**: `add` for f16/bf16 scalars and SIMD packed types
- **9-7-4-2-sub**: `sub` for f16/bf16 scalars and SIMD packed types
- **9-7-4-3-mul**: `mul` for f16/bf16 scalars and SIMD packed types
- **9-7-4-4-fma**: `fma` with relu activation and out-of-bounds (oob) modifier
- **9-7-4-5-to-9-7-4-6**: `neg` (negate) and `abs` (absolute value)
- **9-7-4-7-min**: `min` with NaN propagation and xorsign.abs support
- **9-7-4-8-max**: `max` with NaN propagation and xorsign.abs support
- **9-7-4-9-tanh**: `tanh.approx` for f16/bf16 and SIMD variants
- **9-7-4-10-ex2**: `ex2.approx` (base-2 exponential) for f16/bf16 and SIMD variants
