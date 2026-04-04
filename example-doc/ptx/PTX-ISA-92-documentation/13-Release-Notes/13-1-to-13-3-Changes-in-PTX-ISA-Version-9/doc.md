## 13.1. [Changes in PTX ISA Version 9.2](#changes-in-ptx-isa-version-9-2)

New Features

PTX ISA version 9.2 introduces the following new features:

* Adds support for `.u8x4` and `.s8x4` instruction types for `add`, `sub`, `min`, `max`,
  `neg` instructions.
* Adds support for `add.sat.{u16x2/s16x2/u32}` instruction.
* Adds support for `.b128` type for `st.async` instruction.
* Adds support for `.ignore_oob` qualifier for `cp.async.bulk` instruction.
* Adds support for `.bf16x2` destination type for `cvt` instruction with `.e4m3x2`,
  `.e5m2x2`, `.e3m2x2`, `.e2m3x2`, `.e2m1x2` source types.

Semantic Changes and Clarifications

* For `wgmma.mma_async` instruction with `.atype` and `.btype` as `.e4m3`/`.e5m2` and `.dtype`
  as `.f32`, current implementation does accumulation at higher than half precision but lower than single
  precision.

None.

---

## 13.2. [Changes in PTX ISA Version 9.1](#changes-in-ptx-isa-version-9-1)

New Features

PTX ISA version 9.1 introduces the following new features:

* Adds support for `.volatile` qualifier with `.local` state space for `ld` and
  `st` instructions.
* Adds support for `.f16x2` and `.bf16x2` source types for `cvt` instruction
  with destination types `.e2m1x2`, `.e2m3x2`, `.e3m2x2`, `.e4m3x2`, `.e5m2x2`.
* Adds support for `.scale_vec::4X` with `.ue8m0` as `.stype` with `.kind::mxf4nvf4` for
  `mma`/`mma.sp` instructions.
* Adds support for `.s2f6x2` instruction type for `cvt` instruction.
* Adds support for `multimem.cp.async.bulk` and `multimem.cp.reduce.async.bulk` instructions.

Semantic Changes and Clarifications

None.

---

## 13.3. [Changes in PTX ISA Version 9.0](#changes-in-ptx-isa-version-9-0)

New Features

PTX ISA version 9.0 introduces the following new features:

* Adds support for `sm_88` target architecture.
* Adds support for `sm_110` target architecture.
* Adds support for target `sm_110f` that supports family-specific features.
* Adds support for target `sm_110a` that supports architecture-specific features.
* Adds support for pragma `enable_smem_spilling` that is used to enable shared
  memory spilling for a function.
* Adds support for pragma `frequency` that is used to specify the execution frequency of a basic
  block.
* Adds support for directive `.blocksareclusters` that is used to specify that CUDA thread blocks
  are mapped to clusters.
* Extends `size` operand of `st.bulk` instruction to support 32-bit length.
* Adds support for performance-tuning directives `.abi_preserve` and `.abi_preserve_control`
  that are used to specify the number of data and control registers that should be preserved by the
  callers of a function.

Notes

* Targets `sm_{101,101f,101a}` are renamed to targets `sm_{110,110f,110a}` from PTX ISA version 9.0.

Semantic Changes and Clarifications

* All `tcgen05` instructions(`tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`,
  `tcgen05.cp`, `tcgen05.shift`, `tcgen05.mma`, `tcgen05.mma.sp`, `tcgen05.mma.ws, tcgen05.mma.ws.sp`,
  `tcgen05.commit`) within a kernel must specify the same value for the `.cta_group` qualifier.
