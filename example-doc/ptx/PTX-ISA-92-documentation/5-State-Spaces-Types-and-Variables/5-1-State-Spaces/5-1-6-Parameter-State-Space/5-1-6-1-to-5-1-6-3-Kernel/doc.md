#### 5.1.6.1. [Kernel Function Parameters](#kernel-function-parameters)

Each kernel function definition includes an optional list of parameters. These parameters are
addressable, read-only variables declared in the `.param` state space. Values passed from the host
to the kernel are accessed through these parameter variables using `ld.param` instructions. The
kernel parameter variables are shared across all CTAs from all clusters within a grid.

The address of a kernel parameter may be moved into a register using the `mov` instruction. The
resulting address is in the `.param` state space and is accessed using `ld.param` instructions.

Example

```
.entry foo ( .param .b32 N, .param .align 8 .b8 buffer[64] )
{
    .reg .u32 %n;
    .reg .f64 %d;

    ld.param.u32 %n, [N];
    ld.param.f64 %d, [buffer];
    ...
```

Example

```
.entry bar ( .param .b32 len )
{
    .reg .u32 %ptr, %n;

    mov.u32      %ptr, len;
    ld.param.u32 %n, [%ptr];
    ...
```

Kernel function parameters may represent normal data values, or they may hold addresses to objects
in constant, global, local, or shared state spaces. In the case of pointers, the compiler and
runtime system need information about which parameters are pointers, and to which state space they
point. Kernel parameter attribute directives are used to provide this information at the PTX
level. See [Kernel Function Parameter Attributes](#kernel-function-parameter-attributes)
for a description of kernel parameter attribute
directives.

Note

The current implementation does not allow creation of generic pointers to constant variables
(`cvta.const`) in programs that have pointers to constant buffers passed as kernel parameters.

---

#### 5.1.6.2. [Kernel Function Parameter Attributes](#kernel-function-parameter-attributes)

Kernel function parameters may be declared with an optional .ptr attribute to indicate that a
parameter is a pointer to memory, and also indicate the state space and alignment of the memory
being pointed to. [Kernel Parameter Attribute: .ptr](#kernel-parameter-attribute-ptr)
describes the `.ptr` kernel parameter attribute.

---

#### 5.1.6.3. [Kernel Parameter Attribute: `.ptr`](#kernel-parameter-attribute-ptr)

`.ptr`

Kernel parameter alignment attribute.

Syntax

```
.param .type .ptr .space .align N  varname
.param .type .ptr        .align N  varname

.space = { .const, .global, .local, .shared };
```

Description

Used to specify the state space and, optionally, the alignment of memory pointed to by a pointer
type kernel parameter. The alignment value *N*, if present, must be a power of two. If no state
space is specified, the pointer is assumed to be a generic address pointing to one of const, global,
local, or shared memory. If no alignment is specified, the memory pointed to is assumed to be
aligned to a 4 byte boundary.

Spaces between `.ptr`, `.space`, and `.align` may be eliminated to improve readability.

PTX ISA Notes

* Introduced in PTX ISA version 2.2.
* Support for generic addressing of .const space added in PTX ISA version 3.1.

Target ISA Notes

* Supported on all target architectures.

Examples

```
.entry foo ( .param .u32 param1,
             .param .u32 .ptr.global.align 16 param2,
             .param .u32 .ptr.const.align 8 param3,
             .param .u32 .ptr.align 16 param4  // generic address
                                               // pointer
) { .. }
```
