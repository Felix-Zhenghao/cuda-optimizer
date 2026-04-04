# 9.7.4.5--9.7.4.6 Half Precision Floating Point Instructions: neg, abs

Documents `neg` (arithmetic negate) and `abs` (absolute value) for f16, f16x2, bf16, and bf16x2 types. Both support SIMD packed parallel operations. flush-to-zero available for f16 variants. bf16 support requires sm_80+. NaN inputs yield unspecified NaN.
