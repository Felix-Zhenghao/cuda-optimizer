## 13.14. [Changes in PTX ISA Version 7.7](#changes-in-ptx-isa-version-7-7)

New Features

PTX ISA version 7.7 introduces the following new features:

* Extends `isspacep` and `cvta` instructions to include the `.param` state space for kernel
  function parameters.

Semantic Changes and Clarifications

None.

---

## 13.15. [Changes in PTX ISA Version 7.6](#changes-in-ptx-isa-version-7-6)

New Features

PTX ISA version 7.6 introduces the following new features:

* Support for `szext` instruction which performs sign-extension or zero-extension on a specified
  value.
* Support for `bmsk` instruction which creates a bitmask of the specified width starting at the
  specified bit position.
* Support for special registers `%reserved_smem_offset_begin`, `%reserved_smem_offset_end`,
  `%reserved_smem_offset_cap`, `%reserved_smem_offset<2>`.

Semantic Changes and Clarifications

None.

---

## 13.16. [Changes in PTX ISA Version 7.5](#changes-in-ptx-isa-version-7-5)

New Features

PTX ISA version 7.5 introduces the following new features:

* Debug information enhancements to support label difference and negative values in the `.section`
  debugging directive.
* Support for `ignore-src` operand on `cp.async` instruction.
* Extensions to the memory consistency model to introduce the following new concepts:

  > + A *memory proxy* as an abstract label for different methods of memory access.
  > + Virtual aliases as distinct memory addresses accessing the same physical memory location.
* Support for new `fence.proxy` and `membar.proxy` instructions to allow synchronization of
  memory accesses performed via virtual aliases.

Semantic Changes and Clarifications

None.

---

## 13.17. [Changes in PTX ISA Version 7.4](#changes-in-ptx-isa-version-7-4)

New Features

PTX ISA version 7.4 introduces the following new features:

* Support for `sm_87` target architecture.
* Support for `.level::eviction_priority` qualifier which allows specifying cache eviction
  priority hints on `ld`, `ld.global.nc`, `st`, and `prefetch` instructions.
* Support for `.level::prefetch_size` qualifier which allows specifying data prefetch hints on
  `ld` and `cp.async` instructions.
* Support for `createpolicy` instruction which allows construction of different types of cache
  eviction policies.
* Support for `.level::cache_hint` qualifier which allows the use of cache eviction policies with
  `ld`, `ld.global.nc`, `st`, `atom`, `red` and `cp.async` instructions.
* Support for `applypriority` and `discard` operations on cached data.

Semantic Changes and Clarifications

None.

---

## 13.18. [Changes in PTX ISA Version 7.3](#changes-in-ptx-isa-version-7-3)

New Features

PTX ISA version 7.3 introduces the following new features:

* Extends `mask()` operator used in initializers to also support integer constant expression.
* Adds support for stack manpulation instructions that allow manipulating stack using `stacksave`
  and `stackrestore` instructions and allocation of per-thread stack using `alloca`
  instruction.

Semantic Changes and Clarifications

The unimplemented version of `alloca` from the older PTX ISA specification has been replaced with
new stack manipulation instructions in PTX ISA version 7.3.

---

## 13.19. [Changes in PTX ISA Version 7.2](#changes-in-ptx-isa-version-7-2)

New Features

PTX ISA version 7.2 introduces the following new features:

* Enhances `.loc` directive to represent inline function information.
* Adds support to define labels inside the debug sections.
* Extends `min` and `max` instructions to support `.xorsign` and `.abs` modifiers.

Semantic Changes and Clarifications

None.

---

## 13.20. [Changes in PTX ISA Version 7.1](#changes-in-ptx-isa-version-7-1)

New Features

PTX ISA version 7.1 introduces the following new features:

* Support for `sm_86` target architecture.
* Adds a new operator, `mask()`, to extract a specific byte from variable’s address used in
  initializers.
* Extends `tex` and `tld4` instructions to return an optional predicate that indicates if data
  at specified coordinates is resident in memory.
* Extends single-bit `wmma` and `mma` instructions to support `.and` operation.
* Extends `mma` instruction to support `.sp` modifier that allows matrix multiply-accumulate
  operation when input matrix A is sparse.
* Extends `mbarrier.test_wait` instruction to test the completion of specific phase parity.

Semantic Changes and Clarifications

None.

---

## 13.21. [Changes in PTX ISA Version 7.0](#changes-in-ptx-isa-version-7-0)

New Features

PTX ISA version 7.0 introduces the following new features:

* Support for `sm_80` target architecture.
* Adds support for asynchronous copy instructions that allow copying of data asynchronously from one
  state space to another.
* Adds support for `mbarrier` instructions that allow creation of *mbarrier objects* in memory and
  use of these objects to synchronize threads and asynchronous copy operations initiated by threads.
* Adds support for `redux.sync` instruction which allows reduction operation across threads in a
  warp.
* Adds support for new alternate floating-point data formats `.bf16` and `.tf32`.
* Extends `wmma` instruction to support `.f64` type with shape `.m8n8k4`.
* Extends `wmma` instruction to support `.bf16` data format.
* Extends `wmma` instruction to support `.tf32` data format with shape `.m16n16k8`.
* Extends `mma` instruction to support `.f64` type with shape `.m8n8k4`.
* Extends `mma` instruction to support `.bf16` and `.tf32` data formats with shape
  `.m16n8k8`.
* Extends `mma` instruction to support new shapes `.m8n8k128`, `.m16n8k4`, `.m16n8k16`,
  `.m16n8k32`, `.m16n8k64`, `.m16n8k128` and `.m16n8k256`.
* Extends `abs` and `neg` instructions to support `.bf16` and `.bf16x2` data formats.
* Extends `min` and `max` instructions to support `.NaN` modifier and `.f16`, `.f16x2`,
  `.bf16` and `.bf16x2` data formats.
* Extends `fma` instruction to support `.relu` saturation mode and `.bf16` and `.bf16x2`
  data formats.
* Extends `cvt` instruction to support `.relu` saturation mode and `.f16`, `.f16x2`,
  `.bf16`, `.bf16x2` and `.tf32` destination formats.
* Adds support for `tanh` instruction that computes hyperbolic-tangent.
* Extends `ex2` instruction to support `.f16` and `.f16x2` types.

Semantic Changes and Clarifications

None.

---

## 13.22. [Changes in PTX ISA Version 6.5](#changes-in-ptx-isa-version-6-5)

New Features

PTX ISA version 6.5 introduces the following new features:

* Adds support for integer destination types for half precision comparison instruction `set`.
* Extends `abs` instruction to support `.f16` and `.f16x2` types.
* Adds support for `cvt.pack` instruction which allows converting two integer values and packing
  the results together.
* Adds new shapes `.m16n8k8`, `.m8n8k16` and `.m8n8k32` on the `mma` instruction.
* Adds support for `ldmatrix` instruction which loads one or more matrices from shared memory for
  `mma` instruction.

Removed Features

PTX ISA version 6.5 removes the following features:

* Support for `.satfinite` qualifier on floating point `wmma.mma` instruction has been
  removed. This support was deprecated since PTX ISA version 6.4.

Semantic Changes and Clarifications

None.

---

## 13.23. [Changes in PTX ISA Version 6.4](#changes-in-ptx-isa-version-6-4)

New Features

PTX ISA version 6.4 introduces the following new features:

* Adds support for `.noreturn` directive which can be used to indicate a function does not return
  to it’s caller function.
* Adds support for `mma` instruction which allows performing matrix multiply-and-accumulate
  operation.

Deprecated Features

PTX ISA version 6.4 deprecates the following features:

* Support for `.satfinite` qualifier on floating point `wmma.mma` instruction.

Removed Features

PTX ISA version 6.4 removes the following features:

* Support for `shfl` and `vote` instructions without the `.sync` qualifier has been removed
  for `.target``sm_70` and higher. This support was deprecated since PTX ISA version 6.0 as
  documented in PTX ISA version 6.2.

Semantic Changes and Clarifications

* Clarified that resolving references of a `.weak` symbol considers only `.weak` or `.visible`
  symbols with the same name and does not consider local symbols with the same name.
* Clarified that in `cvt` instruction, modifier `.ftz` can only be specified when either
  `.atype` or `.dtype` is `.f32`.

---

## 13.24. [Changes in PTX ISA Version 6.3](#changes-in-ptx-isa-version-6-3)

New Features

PTX ISA version 6.3 introduces the following new features:

* Support for `sm_75` target architecture.
* Adds support for a new instruction `nanosleep` that suspends a thread for a specified duration.
* Adds support for `.alias` directive which allows definining alias to function symbol.
* Extends `atom` instruction to perform `.f16` addition operation and `.cas.b16` operation.
* Extends `red` instruction to perform `.f16` addition operation.
* The `wmma` instructions are extended to support multiplicand matrices of type `.s8`, `.u8`,
  `.s4`, `.u4`, `.b1` and accumulator matrices of type `.s32`.

Semantic Changes and Clarifications

* Introduced the mandatory `.aligned` qualifier for all `wmma` instructions.
* Specified the alignment required for the base address and stride parameters passed to
  `wmma.load` and `wmma.store`.
* Clarified that layout of fragment returned by `wmma` operation is architecture dependent and
  passing `wmma` fragments around functions compiled for different link compatible SM
  architectures may not work as expected.
* Clarified that atomicity for `{atom/red}.f16x2}` operations is guranteed separately for each of
  the two `.f16` elements but not guranteed to be atomic as single 32-bit access.

---

## 13.25. [Changes in PTX ISA Version 6.2](#changes-in-ptx-isa-version-6-2)

New Features

PTX ISA version 6.2 introduces the following new features:

* A new instruction `activemask` for querying active threads in a warp.
* Extends atomic and reduction instructions to perform `.f16x2` addition operation with mandatory
  `.noftz` qualifier.

Deprecated Features

PTX ISA version 6.2 deprecates the following features:

* The use of `shfl` and `vote` instructions without the `.sync` is deprecated retrospectively
  from PTX ISA version 6.0, which introduced the `sm_70` architecture that implements
  [Independent Thread Scheduling](#independent-thread-scheduling).

Semantic Changes and Clarifications

* Clarified that `wmma` instructions can be used in conditionally executed code only if it is
  known that all threads in the warp evaluate the condition identically, otherwise behavior is
  undefined.
* In the memory consistency model, the definition of *morally strong operations* was updated to
  exclude fences from the requirement of *complete overlap* since fences do not access memory.

---

## 13.26. [Changes in PTX ISA Version 6.1](#changes-in-ptx-isa-version-6-1)

New Features

PTX ISA version 6.1 introduces the following new features:

* Support for `sm_72` target architecture.
* Support for new matrix shapes `32x8x16` and `8x32x16` in `wmma` instruction.

Semantic Changes and Clarifications

None.

---

## 13.27. [Changes in PTX ISA Version 6.0](#changes-in-ptx-isa-version-6-0)

New Features

PTX ISA version 6.0 introduces the following new features:

* Support for `sm_70` target architecture.
* Specifies the memory consistency model for programs running on `sm_70` and later architectures.
* Various extensions to memory instructions to specify memory synchronization semantics and scopes
  at which such synchronization can be observed.
* New instruction `wmma` for matrix operations which allows loading matrices from memory,
  performing multiply-and-accumulate on them and storing result in memory.
* Support for new `barrier` instruction.
* Extends `neg` instruction to support `.f16` and `.f16x2` types.
* A new instruction `fns` which allows finding n-th set bit in integer.
* A new instruction `bar.warp.sync` which allows synchronizing threads in warp.
* Extends `vote` and `shfl` instructions with `.sync` modifier which waits for specified
  threads before executing the `vote` and `shfl` operation respectively.
* A new instruction `match.sync` which allows broadcasting and comparing a value across threads in
  warp.
* A new instruction `brx.idx` which allows branching to a label indexed from list of potential
  targets.
* Support for unsized array parameter for `.func` which can be used to implement variadic
  functions.
* Support for `.b16` integer type in dwarf-lines.
* Support for taking address of device function return parameters using `mov` instruction.

Semantic Changes and Clarifications

* Semantics of `bar` instruction were updated to indicate that executing thread waits for other
  non-exited threads from it’s warp.
* Support for indirect branch introduced in PTX 2.1 which was unimplemented has been removed from
  the spec.
* Support for taking address of labels, using labels in initializers which was unimplemented has
  been removed from the spec.
* Support for variadic functions which was unimplemented has been removed from the spec.

---

## 13.28. [Changes in PTX ISA Version 5.0](#changes-in-ptx-isa-version-5-0)

New Features

PTX ISA version 5.0 introduces the following new features:

* Support for `sm_60`, `sm_61`, `sm_62` target architecture.
* Extends atomic and reduction instructions to perform double-precision add operation.
* Extends atomic and reduction instructions to specify `scope` modifier.
* A new `.common` directive to permit linking multiple object files containing declarations of the
  same symbol with different size.
* A new `dp4a` instruction which allows 4-way dot product with accumulate operation.
* A new `dp2a` instruction which allows 2-way dot product with accumulate operation.
* Support for special register `%clock_hi`.

Semantic Changes and Clarifications

Semantics of cache modifiers on `ld` and `st` instructions were clarified to reflect cache
operations are treated as performance hint only and do not change memory consistency behavior of the
program.

Semantics of `volatile` operations on `ld` and `st` instructions were clarified to reflect how
`volatile` operations are handled by optimizing compiler.
