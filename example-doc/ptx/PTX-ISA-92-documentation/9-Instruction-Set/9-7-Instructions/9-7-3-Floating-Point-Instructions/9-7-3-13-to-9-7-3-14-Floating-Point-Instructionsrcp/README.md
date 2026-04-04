# 9.7.3.13--9.7.3.14 Floating Point Instructions: rcp

Documents PTX `rcp` (reciprocal) in two variants: `rcp.approx` (fast approximation, max 1 ULP error) and `rcp.rnd` (IEEE 754 compliant). Also covers `rcp.approx.ftz.f64` for fast gross double-precision reciprocal using only the upper 32 bits of the mantissa.
