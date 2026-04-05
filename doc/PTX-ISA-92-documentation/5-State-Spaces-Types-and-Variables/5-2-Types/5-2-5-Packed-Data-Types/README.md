# Packed-Data-Types

Covers PTX packed data types across three categories: packed floating-point types (`.f16x2`, `.bf16x2`, `.e4m3x2`, etc., up to 4-element variants), packed integer types (`.u16x2`, `.s16x2`, `.u8x4`, `.s8x4` stored in `.b32`), and packed fixed-point `.s2f6x2`. Only `.f16x2` is a fundamental type; others are instruction types requiring bit-size register operands.
