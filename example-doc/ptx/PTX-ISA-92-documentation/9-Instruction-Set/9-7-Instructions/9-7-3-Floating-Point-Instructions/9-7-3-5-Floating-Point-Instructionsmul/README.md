# 9.7.3.5 Floating Point Instructions: mul

Documents PTX floating-point `mul` for f32, f32x2, and f64 types. Supports rounding modes (rn/rz/rm/rp), flush-to-zero, and saturation. Mul/add sequences without explicit rounding may be fused into FMA by the optimizer. The f32x2 SIMD variant requires sm_100+.
