### 7.1.1. [Changes from PTX ISA Version 1.x](#changes-from-ptx-isa-version-1-x)

In PTX ISA version 1.x, formal parameters were restricted to .reg state space, and there was no
support for array parameters. Objects such as C structures were flattened and passed or returned
using multiple registers. PTX ISA version 1.x supports multiple return values for this purpose.

Beginning with PTX ISA version 2.0, formal parameters may be in either `.reg` or `.param` state
space, and `.param` space parameters support arrays. For targets `sm_20` or higher, PTX
restricts functions to a single return value, and a `.param` byte array should be used to return
objects that do not fit into a register. PTX continues to support multiple return registers for
`sm_1x` targets.

Note

PTX implements a stack-based ABI only for targets `sm_20` or higher.

PTX ISA versions prior to 3.0 permitted variables in `.reg` and `.local` state spaces to be
defined at module scope. When compiling to use the ABI, PTX ISA version 3.0 and later disallows
module-scoped `.reg` and `.local` variables and restricts their use to within function
scope. When compiling without use of the ABI, module-scoped `.reg` and `.local` variables are
supported as before. When compiling legacy PTX code (ISA versions prior to 3.0) containing
module-scoped `.reg` or `.local` variables, the compiler silently disables use of the ABI.

---

## 7.2. [Variadic Functions](#variadic-functions)

Note

**Support for variadic functions which was unimplemented has been removed from the spec.**

PTX version 6.0 supports passing unsized array parameter to a function which can be used to
implement variadic functions.

Refer to [Kernel and Function Directives: .func](#kernel-and-function-directives-func) for details

---

## 7.3. [Alloca](#alloca)

PTX provides `alloca` instruction for allocating storage at runtime on the per-thread local memory
stack. The allocated stack memory can be accessed with `ld.local` and `st.local` instructions
using the pointer returned by `alloca`.

In order to facilitate deallocation of memory allocated with `alloca`, PTX provides two additional
instructions: `stacksave` which allows reading the value of stack pointer in a local variable, and
`stackrestore` which can restore the stack pointer with the saved value.

`alloca`, `stacksave`, and `stackrestore` instructions are described in
[Stack Manipulation Instructions](#stack-manipulation-instructions).
