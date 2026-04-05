### 5.4.1. [Variable Declarations](#variable-declarations)

All storage for data is specified with variable declarations. Every variable must reside in one of
the state spaces enumerated in the previous section.

A variable declaration names the space in which the variable resides, its type and size, its name,
an optional array size, an optional initializer, and an optional fixed address for the variable.

Predicate variables may only be declared in the register state space.

Examples

```
.global .u32 loc;
.reg    .s32 i;
.const  .f32 bias[] = {-1.0, 1.0};
.global .u8  bg[4] = {0, 0, 0, 0};
.reg    .v4 .f32 accel;
.reg    .pred p, q, r;
```

---

### 5.4.2. [Vectors](#vectors)

Limited-length vector types are supported. Vectors of length 2 and 4 of any non-predicate
fundamental type can be declared by prefixing the type with `.v2` or `.v4`. Vectors must be
based on a fundamental type, and they may reside in the register space. Vectors cannot exceed
128-bits in length; for example, `.v4 .f64` is not allowed. Three-element vectors may be
handled by using a `.v4` vector, where the fourth element provides padding. This is a common case
for three-dimensional grids, textures, etc.

Examples

```
.global .v4 .f32 V;   // a length-4 vector of floats
.shared .v2 .u16 uv;  // a length-2 vector of unsigned ints
.global .v4 .b8  v;   // a length-4 vector of bytes
```

By default, vector variables are aligned to a multiple of their overall size (vector length times
base-type size), to enable vector load and store instructions which require addresses aligned to a
multiple of the access size.

---

### 5.4.3. [Array Declarations](#array-declarations)

Array declarations are provided to allow the programmer to reserve space. To declare an array, the
variable name is followed with dimensional declarations similar to fixed-size array declarations
in C. The size of each dimension is a constant expression.

Examples

```
.local  .u16 kernel[19][19];
.shared .u8  mailbox[128];
```

The size of the array specifies how many elements should be reserved. For the declaration of array
*kernel* above, 19\*19 = 361 halfwords are reserved, for a total of 722 bytes.

When declared with an initializer, the first dimension of the array may be omitted. The size of the
first array dimension is determined by the number of elements in the array initializer.

Examples

```
.global .u32 index[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
.global .s32 offset[][2] = { {-1, 0}, {0, -1}, {1, 0}, {0, 1} };
```

Array *index* has eight elements, and array *offset* is a 4x2 array.
