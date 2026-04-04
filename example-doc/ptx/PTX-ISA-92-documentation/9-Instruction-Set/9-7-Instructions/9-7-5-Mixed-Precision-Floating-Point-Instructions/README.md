# 9.7.5 Mixed Precision Floating Point Instructions

Documents mixed-precision PTX instructions that convert f16/bf16 inputs to f32 before computing: `add` (addition with f16/bf16-to-f32 promotion), `sub` (subtraction with promotion), and `fma` (fused multiply-add with promoted inputs). All support IEEE rounding modes and saturation. Requires sm_100 or higher. Introduced in PTX ISA 8.6.
