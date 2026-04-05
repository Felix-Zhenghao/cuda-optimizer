### 5.2.1. [Fundamental Types](#fundamental-types)

In PTX, the fundamental types reflect the native data types supported by the target architectures. A
fundamental type specifies both a basic type and a size. Register variables are always of a
fundamental type, and instructions operate on these types. The same type-size specifiers are used
for both variable definitions and for typing instructions, so their names are intentionally short.

[Table 8](#fundamental-types-fundamental-type-specifiers) lists the fundamental type specifiers for
each basic type:

Table 8 Fundamental Type Specifiers

| Basic Type | Fundamental Type Specifiers |
| --- | --- |
| Signed integer | `.s8`, `.s16`, `.s32`, `.s64` |
| Unsigned integer | `.u8`, `.u16`, `.u32`, `.u64` |
| Floating-point | `.f16`, `.f16x2`, `.f32`, `.f64` |
| Bits (untyped) | `.b8`, `.b16`, `.b32`, `.b64`, `.b128` |
| Predicate | `.pred` |

Most instructions have one or more type specifiers, needed to fully specify instruction
behavior. Operand types and sizes are checked against instruction types for compatibility.

Two fundamental types are compatible if they have the same basic type and are the same size. Signed
and unsigned integer types are compatible if they have the same size. The bit-size type is
compatible with any fundamental type having the same size.

In principle, all variables (aside from predicates) could be declared using only bit-size types, but
typed variables enhance program readability and allow for better operand type checking.

---

### 5.2.2. [Restricted Use of Sub-Word Sizes](#restricted-use-of-sub-word-sizes)

The `.u8`, `.s8`, and `.b8` instruction types are restricted to `ld`, `st`, `add`, `sub`,
`min`, `max`, `neg` and `cvt` instructions. The `.f16` floating-point type is allowed
in half precision floating point instructions and texture fetch instructions.
The `.f16x2` floating point type is allowed only in half precision
floating point arithmetic instructions and texture fetch instructions.

For convenience, `ld`, `st`, and `cvt` instructions permit source and destination data
operands to be wider than the instruction-type size, so that narrow values may be loaded, stored,
and converted using regular-width registers. For example, 8-bit or 16-bit values may be held
directly in 32-bit or 64-bit registers when being loaded, stored, or converted to other types and
sizes.
