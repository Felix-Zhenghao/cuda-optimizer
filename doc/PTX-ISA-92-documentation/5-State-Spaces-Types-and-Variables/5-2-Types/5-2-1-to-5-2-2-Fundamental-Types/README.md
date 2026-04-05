# Fundamental-Types

Merged from: 5-2-1-Fundamental-Types to 5-2-2-Restricted-Use-of-Sub-Word-Sizes

Defines PTX fundamental types: signed/unsigned integers (`.s8`–`.s64`, `.u8`–`.u64`), floating-point (`.f16`, `.f16x2`, `.f32`, `.f64`), untyped bit-size (`.b8`–`.b128`), and predicate (`.pred`). Types of equal size and compatible basic type are interchangeable. Sub-word types `.u8`, `.s8`, `.b8` are restricted to `ld`, `st`, `cvt`, and arithmetic instructions.
