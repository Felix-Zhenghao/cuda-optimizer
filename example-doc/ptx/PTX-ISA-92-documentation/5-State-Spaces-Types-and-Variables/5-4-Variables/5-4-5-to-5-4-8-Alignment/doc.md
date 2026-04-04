### 5.4.5. [Alignment](#alignment)

Byte alignment of storage for all addressable variables can be specified in the variable
declaration. Alignment is specified using an optional `.align` *byte-count* specifier immediately
following the state-space specifier. The variable will be aligned to an address which is an integer
multiple of byte-count. The alignment value byte-count must be a power of two. For arrays, alignment
specifies the address alignment for the starting address of the entire array, not for individual
elements.

The default alignment for scalar and array variables is to a multiple of the base-type size. The
default alignment for vector variables is to a multiple of the overall vector size.

Examples

```
 // allocate array at 4-byte aligned address.  Elements are bytes.
.const .align 4 .b8 bar[8] = {0,0,0,0,2,0,0,0};
```

Note that all PTX instructions that access memory require that the address be aligned to a multiple
of the access size. The access size of a memory instruction is the total number of bytes accessed in
memory. For example, the access size of `ld.v4.b32` is 16 bytes, while the access size of
`atom.f16x2` is 4 bytes.

---

### 5.4.6. [Parameterized Variable Names](#parameterized-variable-names)

Since PTX supports virtual registers, it is quite common for a compiler frontend to generate a large
number of register names. Rather than require explicit declaration of every name, PTX supports a
syntax for creating a set of variables having a common prefix string appended with integer suffixes.

For example, suppose a program uses a large number, say one hundred, of `.b32` variables, named
`%r0`, `%r1`, ..., `%r99`. These 100 register variables can be declared as follows:

```
.reg .b32 %r<100>;    // declare %r0, %r1, ..., %r99
```

This shorthand syntax may be used with any of the fundamental types and with any state space, and
may be preceded by an alignment specifier. Array variables cannot be declared this way, nor are
initializers permitted.

---

### 5.4.7. [Variable Attributes](#variable-attributes)

Variables may be declared with an optional `.attribute` directive which allows specifying special
attributes of variables. Keyword `.attribute` is followed by attribute specification inside
parenthesis. Multiple attributes are separated by comma.

[Variable and Function Attribute Directive: .attribute](#variable-and-function-attribute-directive-attribute) describes the `.attribute`
directive.

---

### 5.4.8. [Variable and Function Attribute Directive: `.attribute`](#variable-and-function-attribute-directive-attribute)

`.attribute`

Variable and function attributes

Description

Used to specify special attributes of a variable or a function.

The following attributes are supported.

`.managed`
:   `.managed` attribute specifies that variable will be allocated at a location in unified virtual
    memory environment where host and other devices in the system can reference the variable
    directly. This attribute can only be used with variables in .global state space. See the *CUDA
    UVM-Lite Programming Guide* for details.

`.unified`
:   `.unified` attribute specifies that function has the same memory address on the host and on
    other devices in the system. Integer constants `uuid1` and `uuid2` respectively specify upper
    and lower 64 bits of the unique identifier associated with the function or the variable. This
    attribute can only be used on device functions or on variables in the `.global` state
    space. Variables with `.unified` attribute are read-only and must be loaded by specifying
    `.unified` qualifier on the address operand of `ld` instruction, otherwise the behavior is
    undefined.

PTX ISA Notes

* Introduced in PTX ISA version 4.0.
* Support for function attributes introduced in PTX ISA version 8.0.

Target ISA Notes

* `.managed` attribute requires `sm_30` or higher.
* `.unified` attribute requires `sm_90` or higher.

Examples

```
.global .attribute(.managed) .s32 g;
.global .attribute(.managed) .u64 x;

.global .attribute(.unified(19,95)) .f32 f;

.func .attribute(.unified(0xAB, 0xCD)) bar() { ... }
```
