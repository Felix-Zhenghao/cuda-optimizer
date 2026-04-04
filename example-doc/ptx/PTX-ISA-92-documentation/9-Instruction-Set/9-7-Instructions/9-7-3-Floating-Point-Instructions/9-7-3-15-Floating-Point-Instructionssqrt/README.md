# 9.7.3.15 Floating Point Instructions: sqrt

Documents PTX `sqrt` (square root) for f32 and f64. Provides `sqrt.approx.f32` (fast, max relative error 2^-23) and `sqrt.rnd` (IEEE 754 compliant with explicit rounding). Covers corner cases for negative inputs, zero, infinity, and NaN.
