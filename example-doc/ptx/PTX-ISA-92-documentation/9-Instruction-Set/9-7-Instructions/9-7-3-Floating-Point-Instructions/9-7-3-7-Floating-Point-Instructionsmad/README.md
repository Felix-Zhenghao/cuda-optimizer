# 9.7.3.7 Floating Point Instructions: mad

Documents PTX `mad` (multiply-add) for f32 and f64. On sm_20+, `mad` is equivalent to `fma` with full IEEE 754 precision. On sm_1x, mad.f32 uses double-precision intermediate with truncated mantissa. Requires explicit rounding for sm_20+ targets.
