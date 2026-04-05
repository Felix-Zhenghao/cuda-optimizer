# 13.39. Changes in PTX ISA Version 2.0

PTX ISA 2.0 introduces IEEE 754 compliance for sm_20 targets: subnormal support for single-precision, .rm/.rp rounding for add/sub/mul, fused multiply-add (fma.f32), IEEE-compliant div/rcp/sqrt, testp/copysign instructions, and major ABI/parameter-passing improvements. Maintains backward compatibility with sm_1x via .ftz modifier.
