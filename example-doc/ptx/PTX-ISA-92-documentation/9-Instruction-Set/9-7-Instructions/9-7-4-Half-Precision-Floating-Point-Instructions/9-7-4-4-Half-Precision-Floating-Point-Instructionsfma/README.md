# 9.7.4.4 Half Precision Floating Point Instructions: fma

Documents half-precision `fma` (fused multiply-add) for f16, f16x2, bf16, and bf16x2. Supports relu saturation (clamp negatives to zero), out-of-bounds NaN modifier, and flush-to-zero for f16. bf16 requires sm_80+; oob modifier requires sm_90+.
