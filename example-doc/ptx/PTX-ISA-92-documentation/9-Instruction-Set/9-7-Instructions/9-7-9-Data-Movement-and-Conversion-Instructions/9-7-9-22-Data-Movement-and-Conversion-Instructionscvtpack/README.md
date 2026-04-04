# 9.7.9.22. Data Movement and Conversion Instructions: cvt.pack

The `cvt.pack` instruction converts two 32-bit integers to a smaller type (.u16/.s16 or sub-byte .u2/.s2/.u4/.s4/.u8/.s8) with saturation and packs results into a 32-bit destination. An optional third operand supplies remaining bits. Useful for quantization and packing operations. Requires sm_72+; sub-byte types require sm_75+. Introduced in PTX ISA 6.5.
