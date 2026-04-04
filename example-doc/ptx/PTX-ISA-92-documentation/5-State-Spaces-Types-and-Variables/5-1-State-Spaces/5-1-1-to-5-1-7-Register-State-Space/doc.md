### 5.1.1. [Register State Space](#register-state-space)

Registers (`.reg` state space) are fast storage locations. The number of registers is limited, and
will vary from platform to platform. When the limit is exceeded, register variables will be spilled
to memory, causing changes in performance. For each architecture, there is a recommended maximum
number of registers to use (see the *CUDA Programming Guide* for details).

Registers may be typed (signed integer, unsigned integer, floating point, predicate) or
untyped. Register size is restricted; aside from predicate registers which are 1-bit, scalar
registers have a width of 8-, 16-, 32-, 64-, or 128-bits, and vector registers have a width of
16-, 32-, 64-, or 128-bits. The most common use of 8-bit registers is with `ld`, `st`, and `cvt`
instructions, or as elements of vector tuples.

Registers differ from the other state spaces in that they are not fully addressable, i.e., it is not
possible to refer to the address of a register. When compiling to use the Application Binary
Interface (ABI), register variables are restricted to function scope and may not be declared at
module scope. When compiling legacy PTX code (ISA versions prior to 3.0) containing module-scoped
`.reg` variables, the compiler silently disables use of the ABI. Registers may have alignment
boundaries required by multi-word loads and stores.

---

### 5.1.2. [Special Register State Space](#special-register-state-space)

The special register (`.sreg`) state space holds predefined, platform-specific registers, such as
grid, cluster, CTA, and thread parameters, clock counters, and performance monitoring registers. All
special registers are predefined.

---

#### 5.1.3.1. [Banked Constant State Space (deprecated)](#banked-constant-state-space-deprecated)

Previous versions of PTX exposed constant memory as a set of eleven 64 KB banks, with explicit bank
numbers required for variable declaration and during access.

Prior to PTX ISA version 2.2, the constant memory was organized into fixed size banks. There were
eleven 64 KB banks, and banks were specified using the `.const[bank]` modifier, where *bank*
ranged from 0 to 10. If no bank number was given, bank zero was assumed.

By convention, bank zero was used for all statically-sized constant variables. The remaining banks
were used to declare *incomplete* constant arrays (as in C, for example), where the size is not
known at compile time. For example, the declaration

```
.extern .const[2] .b32 const_buffer[];
```

resulted in `const_buffer` pointing to the start of constant bank two. This pointer could then be
used to access the entire 64 KB constant bank. Multiple incomplete array variables declared in the
same bank were aliased, with each pointing to the start address of the specified constant bank.

To access data in contant banks 1 through 10, the bank number was required in the state space of the
load instruction. For example, an incomplete array in bank 2 was accessed as follows:

```
.extern .const[2] .b32 const_buffer[];
ld.const[2].b32  %r1, [const_buffer+4]; // load second word
```

In PTX ISA version 2.2, we eliminated explicit banks and replaced the incomplete array
representation of driver-allocated constant buffers with kernel parameter attributes that allow
pointers to constant buffers to be passed as kernel parameters.

---

### 5.1.4. [Global State Space](#global-state-space)

The global (`.global`) state space is memory that is accessible by all threads in a context. It is
the mechanism by which threads in different CTAs, clusters, and grids can communicate. Use
`ld.global`, `st.global`, and `atom.global` to access global variables.

Global variables have an optional variable initializer; global variables with no explicit
initializer are initialized to zero by default.

---

### 5.1.5. [Local State Space](#local-state-space)

The local state space (`.local`) is private memory for each thread to keep its own data. It is
typically standard memory with cache. The size is limited, as it must be allocated on a per-thread
basis. Use `ld.local` and `st.local` to access local variables.

When compiling to use the *Application Binary Interface (ABI)*, `.local` state-space variables
must be declared within function scope and are allocated on the stack. In implementations that do
not support a stack, all local memory variables are stored at fixed addresses, recursive function
calls are not supported, and `.local` variables may be declared at module scope. When compiling
legacy PTX code (ISA versions prior to 3.0) containing module-scoped `.local` variables, the
compiler silently disables use of the ABI.

---

### 5.1.7. [Shared State Space](#shared-state-space)

The shared (`.shared`) state space is a memory that is owned by an executing CTA and is accessible
to the threads of all the CTAs within a cluster. An address in shared memory can be read and written
by any thread in a CTA cluster.

Additional sub-qualifiers `::cta` or `::cluster` can be specified on instructions with
`.shared` state space to indicate whether the address belongs to the shared memory window of the
executing CTA or of any CTA in the cluster respectively. The addresses in the `.shared::cta`
window also fall within the `.shared::cluster` window. If no sub-qualifier is specified with the
`.shared` state space, then it defaults to `::cta`. For example, `ld.shared` is equivalent to
`ld.shared::cta`.

Variables declared in `.shared` state space refer to the memory addresses in the current
CTA. Instruction `mapa` gives the `.shared::cluster` address of the corresponding variable in
another CTA in the cluster.

Shared memory typically has some optimizations to support the sharing. One example is broadcast;
where all threads read from the same address. Another is sequential access from sequential threads.
