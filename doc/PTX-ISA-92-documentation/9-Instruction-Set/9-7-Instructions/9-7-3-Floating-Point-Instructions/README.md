PTX floating-point instructions covering the full range of single and double precision operations.

- **9-7-3-1-to-9-7-3-2**: `testp` (test FP properties) and `copysign` (copy sign bit)
- **9-7-3-3-add**: `add` instruction for f32, f64, and SIMD f32x2
- **9-7-3-4-sub**: `sub` instruction for f32, f64, and SIMD f32x2
- **9-7-3-5-mul**: `mul` instruction for f32, f64, and SIMD f32x2
- **9-7-3-6-fma**: `fma` fused multiply-add for f32, f64, and SIMD f32x2
- **9-7-3-7-mad**: `mad` multiply-add with legacy sm_1x behavior
- **9-7-3-8-div**: `div` with approximate and IEEE-compliant modes
- **9-7-3-9-to-9-7-3-10**: `abs` (absolute value) and `neg` (negate)
- **9-7-3-11-min**: `min` with optional NaN propagation and xorsign.abs
- **9-7-3-12-max**: `max` with optional NaN propagation and xorsign.abs
- **9-7-3-13-to-9-7-3-14-rcp**: `rcp` and `rcp.approx.ftz.f64` reciprocal
- **9-7-3-15-sqrt**: `sqrt` with approximate and IEEE-compliant modes
- **9-7-3-16-to-9-7-3-22**: `rsqrt`, `sin`, `cos`, `lg2`, `ex2`, `tanh` approximations
