### 6.4.2. [Arrays as Operands](#arrays-as-operands)

Arrays of all types can be declared, and the identifier becomes an address constant in the space
where the array is declared. The size of the array is a constant in the program.

Array elements can be accessed using an explicitly calculated byte address, or by indexing into the
array using square-bracket notation. The expression within square brackets is either a constant
integer, a register variable, or a simple *register with constant offset* expression, where the
offset is a constant expression that is either added or subtracted from a register variable. If more
complicated indexing is desired, it must be written as an address calculation prior to use. Examples
are:

```
ld.global.u32  s, a[0];
ld.global.u32  s, a[N-1];
mov.u32        s, a[1];  // move address of a[1] into s
```

---

### 6.4.3. [Vectors as Operands](#vectors-as-operands)

Vector operands can be specified as source and destination operands for instructions. However, when
specified as destination operand, all elements in vector expression must be unique, otherwise behavior
is undefined.
Vectors may also be passed as arguments to called functions.

Vector elements can be extracted from the vector with the suffixes `.x`, `.y`, `.z` and
`.w`, as well as the typical color fields `.r`, `.g`, `.b` and `.a`.

A brace-enclosed list is used for pattern matching to pull apart vectors.

```
.reg .v4 .f32 V;
.reg .f32     a, b, c, d;

mov.v4.f32 {a,b,c,d}, V;
```

Vector loads and stores can be used to implement wide loads and stores, which may improve memory
performance. The registers in the load/store operations can be a vector, or a brace-enclosed list of
similarly typed scalars. Here are examples:

```
ld.global.v4.f32  {a,b,c,d}, [addr+16];
ld.global.v2.u32  V2, [addr+8];
```

Elements in a brace-enclosed vector, say {Ra, Rb, Rc, Rd}, correspond to extracted elements as follows:

```
Ra = V.x = V.r
Rb = V.y = V.g
Rc = V.z = V.b
Rd = V.w = V.a
```

---

### 6.4.4. [Labels and Function Names as Operands](#labels-and-function-names-as-operands)

Labels and function names can be used only in `bra`/`brx.idx` and `call` instructions
respectively. Function names can be used in `mov` instruction to get the address of the function
into a register, for use in an indirect call.

Beginning in PTX ISA version 3.1, the `mov` instruction may be used to take the address of kernel
functions, to be passed to a system call that initiates a kernel launch from the GPU. This feature
is part of the support for CUDA Dynamic Parallelism. See the *CUDA Dynamic Parallelism Programming
Guide* for details.
