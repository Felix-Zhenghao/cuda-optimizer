# 9.7.3.6 Floating Point Instructions: fma

Documents PTX `fma` (fused multiply-add) for f32, f32x2, and f64. Computes a*b+c with no precision loss in intermediate results. Requires explicit rounding mode. f32x2 SIMD variant (PTX 8.6, sm_100+) performs two parallel FMA operations.
