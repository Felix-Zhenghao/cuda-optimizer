### 4.5.1. [Integer Constants](#integer-constants)

Integer constants are 64-bits in size and are either signed or unsigned, i.e., every integer
constant has type `.s64` or `.u64`. The signed/unsigned nature of an integer constant is needed
to correctly evaluate constant expressions containing operations such as division and ordered
comparisons, where the behavior of the operation depends on the operand types. When used in an
instruction or data initialization, each integer constant is converted to the appropriate size based
on the data or instruction type at its use.

Integer literals may be written in decimal, hexadecimal, octal, or binary notation. The syntax
follows that of C. Integer literals may be followed immediately by the letter `U` to indicate that
the literal is unsigned.

```
hexadecimal literal:  0[xX]{hexdigit}+U?
octal literal:        0{octal digit}+U?
binary literal:       0[bB]{bit}+U?
decimal literal       {nonzero-digit}{digit}*U?
```

Integer literals are non-negative and have a type determined by their magnitude and optional type
suffix as follows: literals are signed (`.s64`) unless the value cannot be fully represented in
`.s64` or the unsigned suffix is specified, in which case the literal is unsigned (`.u64`).

The predefined integer constant `WARP_SZ` specifies the number of threads per warp for the target
platform; to date, all target architectures have a `WARP_SZ` value of 32.

---

### 4.5.2. [Floating-Point Constants](#floating-point-constants)

Floating-point constants are represented as 64-bit double-precision values, and all floating-point
constant expressions are evaluated using 64-bit double precision arithmetic. The only exception is
the 32-bit hex notation for expressing an exact single-precision floating-point value; such values
retain their exact 32-bit single-precision value and may not be used in constant expressions. Each
64-bit floating-point constant is converted to the appropriate floating-point size based on the data
or instruction type at its use.

Floating-point literals may be written with an optional decimal point and an optional signed
exponent. Unlike C and C++, there is no suffix letter to specify size; literals are always
represented in 64-bit double-precision format.

PTX includes a second representation of floating-point constants for specifying the exact machine
representation using a hexadecimal constant. To specify IEEE 754 double-precision floating point
values, the constant begins with `0d` or `0D` followed by 16 hex digits. To specify IEEE 754
single-precision floating point values, the constant begins with `0f` or `0F` followed by 8 hex
digits.

```
0[fF]{hexdigit}{8}      // single-precision floating point
0[dD]{hexdigit}{16}     // double-precision floating point
```

Example

```
mov.f32  $f3, 0F3f800000;       //  1.0
```

---

### 4.5.3. [Predicate Constants](#predicate-constants)

In PTX, integer constants may be used as predicates. For predicate-type data initializers and
instruction operands, integer constants are interpreted as in C, i.e., zero values are `False` and
non-zero values are `True`.
