# 5.2.3. Alternate Floating-Point Data Formats

Describes PTX alternate floating-point formats beyond the fundamental types: `bf16` (8-exp, 7-man), `e4m3` (4-exp, 3-man, no inf), `e5m2` (5-exp, 2-man), `tf32` (32-bit reduced precision for matrix ops), `e2m1`/`e2m3`/`e3m2` (4–6 bit sub-byte formats), `ue8m0` (8-bit unsigned, exponent-only), and `ue4m3` (7-bit unsigned). All must be stored in bit-size register types.
