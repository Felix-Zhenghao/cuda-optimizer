# 9.7.9.21. Data Movement and Conversion Instructions: cvt

The `cvt` instruction converts values between types and sizes. Supports conversions among integers (u8-u64, s8-s64), floats (f16, bf16, f32, f64), and specialized formats (tf32, e4m3, e5m2, e2m1, e2m3, e3m2, ue8m0, s2f6). Includes packed conversions (f16x2, bf16x2, etc.), rounding modes (.rn, .rz, .rm, .rp, .rs stochastic), saturation (.sat, .relu, .satfinite), and optional scaling. Introduced in PTX ISA 1.0.
