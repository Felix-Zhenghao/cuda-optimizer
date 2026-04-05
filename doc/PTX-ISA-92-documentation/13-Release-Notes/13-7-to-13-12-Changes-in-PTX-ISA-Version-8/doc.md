## 13.7. [Changes in PTX ISA Version 8.5](#changes-in-ptx-isa-version-8-5)

New Features

PTX ISA version 8.5 introduces the following new features:

* Adds support for `mma.sp::ordered_metadata` instruction.

Semantic Changes and Clarifications

* Values `0b0000`, `0b0101`, `0b1010`, `0b1111` for sparsity metadata (operand `e`)
  of instruction `mma.sp` are invalid and their usage results in undefined behavior.

---

## 13.8. [Changes in PTX ISA Version 8.4](#changes-in-ptx-isa-version-8-4)

New Features

PTX ISA version 8.4 introduces the following new features:

* Extends `ld`, `st` and `atom` instructions with `.b128` type to support `.sys` scope.
* Extends integer `wgmma.mma_async` instruction to support `.u8.s8` and `.s8.u8` as `.atype`
  and `.btype` respectively.
* Extends `mma`, `mma.sp` instructions to support FP8 types `.e4m3` and `.e5m2`.

Semantic Changes and Clarifications

None.

---

## 13.9. [Changes in PTX ISA Version 8.3](#changes-in-ptx-isa-version-8-3)

New Features

PTX ISA version 8.3 introduces the following new features:

* Adds support for pragma `used_bytes_mask` that is used to specify mask for used bytes for a load operation.
* Extends `isspacep`, `cvta.to`, `ld` and `st` instructions to accept `::entry` and `::func`
  sub-qualifiers with `.param` state space qualifier.
* Adds support for `.b128` type on instructions `ld`, `ld.global.nc`, `ldu`, `st`, `mov` and `atom`.
* Add support for instructions `tensormap.replace`, `tensormap.cp_fenceproxy` and support for qualifier
  `.to_proxykind::from_proxykind` on instruction `fence.proxy` to support modifying `tensor-map`.

Semantic Changes and Clarifications

None.

---

## 13.10. [Changes in PTX ISA Version 8.2](#changes-in-ptx-isa-version-8-2)

New Features

PTX ISA version 8.2 introduces the following new features:

* Adds support for `.mmio` qualifier on `ld` and `st` instructions.
* Extends `lop3` instruction to allow predicate destination.
* Extends `multimem.ld_reduce` instruction to support `.acc::f32` qualifer to allow `.f32`
  precision of the intermediate accumulation.
* Extends the asynchronous warpgroup-level matrix multiply-and-accumulate operation
  `wgmma.mma_async` to support `.sp` modifier that allows matrix multiply-accumulate operation
  when input matrix A is sparse.

Semantic Changes and Clarifications

The `.multicast::cluster` qualifier on `cp.async.bulk` and `cp.async.bulk.tensor` instructions
is optimized for target architecture `sm_90a` and may have substantially reduced performance on
other targets and hence `.multicast::cluster` is advised to be used with `sm_90a`.

---

## 13.11. [Changes in PTX ISA Version 8.1](#changes-in-ptx-isa-version-8-1)

New Features

PTX ISA version 8.1 introduces the following new features:

* Adds support for `st.async` and `red.async` instructions for asynchronous store and
  asynchronous reduction operations respectively on shared memory.
* Adds support for `.oob` modifier on half-precision `fma` instruction.
* Adds support for `.satfinite` saturation modifer on `cvt` instruction for `.f16`, `.bf16`
  and `.tf32` formats.
* Extends support for `cvt` with `.e4m3`/`.e5m2` to `sm_89`.
* Extends `atom` and `red` instructions to support vector types.
* Adds support for special register `%aggr_smem_size`.
* Extends `sured` instruction with 64-bit `min`/`max` operations.
* Adds support for increased kernel parameter size of 32764 bytes.
* Adds support for multimem addresses in memory consistency model.
* Adds support for `multimem.ld_reduce`, `multimem.st` and `multimem.red` instructions to
  perform memory operations on multimem addresses.

Semantic Changes and Clarifications

None.

---

## 13.12. [Changes in PTX ISA Version 8.0](#changes-in-ptx-isa-version-8-0)

New Features

PTX ISA version 8.0 introduces the following new features:

* Adds support for target `sm_90a` that supports architecture-specific features.
* Adds support for asynchronous warpgroup-level matrix multiply-and-accumulate operation `wgmma`.
* Extends the asynchronous copy operations with bulk operations that operate on large data,
  including tensor data.
* Introduces packed integer types `.u16x2` and `.s16x2`.
* Extends integer arithmetic instruction `add` to allow packed integer types `.u16x2` and `.s16x2`.
* Extends integer arithmetic instructions `min` and `max` to allow packed integer types
  `.u16x2` and `.s16x2`, as well as saturation modifier `.relu` on `.s16x2` and `.s32`
  types.
* Adds support for special register `%current_graph_exec` that identifies the currently executing
  CUDA device graph.
* Adds support for `elect.sync` instruction.
* Adds support for `.unified` attribute on functions and variables.
* Adds support for `setmaxnreg` instruction.
* Adds support for `.sem` qualifier on `barrier.cluster` instruction.
* Extends the `fence` instruction to allow opcode-specific synchronizaion using `op_restrict`
  qualifier.
* Adds support for `.cluster` scope on `mbarrier.arrive`, `mbarrier.arrive_drop`,
  `mbarrier.test_wait` and `mbarrier.try_wait` operations.
* Adds support for transaction count operations on `mbarrier` objects, specified with
  `.expect_tx` and `.complete_tx` qualifiers.

Semantic Changes and Clarifications

None.
