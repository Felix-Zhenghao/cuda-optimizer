# PTX ISA 9.2 Documentation

NVIDIA Parallel Thread Execution (PTX) Instruction Set Architecture version 9.2 reference documentation.

- **1-Introduction** — PTX virtual ISA overview: motivation, goals, version 9.2 highlights, and documentation structure.
- **2-Programming-Model** — GPU coprocessor model with thread/CTA/cluster/grid hierarchy and multi-level memory spaces.
- **3-PTX-Machine-Model** — SIMT multiprocessor architecture: warp execution, independent thread scheduling, on-chip memory hierarchy.
- **4-Syntax** — PTX source format: ASCII encoding, C preprocessor, directives, instruction statements, constants, and expressions.
- **5-State-Spaces-Types-and-Variables** — Memory state spaces (register, shared, global, etc.), fundamental/alternate types, textures, and variable declarations.
- **6-Instruction-Operands** — Operand type rules, addressing modes, arrays, vectors, labels, type conversions, and rounding modifiers.
- **7-Abstracting-the-ABI** — ABI abstraction: .param space, stack-based calling, variadic support, and alloca for sm_20+.
- **8-Memory-Consistency-Model** — Formal memory model: scopes, proxies, release/acquire patterns, ordering rules, and six consistency axioms.
- **9-Instruction-Set** — Complete instruction reference: arithmetic, FP, data movement, control flow, synchronization, matrix, and tensor ops.
- **10-Special-Registers** — Read-only registers: thread/CTA/grid identity, lane masks, cycle counters, performance monitors, timers.
- **11-Directives** — Module, kernel, function, control flow, performance, debugging, linking, and cluster directives.
- **12-Descriptions-ofpragmaStrings** — Pragma strings: nounroll, used_bytes_mask, enable_smem_spilling, and frequency hints.
- **13-Release-Notes** — Version-by-version changelog from PTX 2.0 through 9.2 with new targets, instructions, and features.
- **14-Notices** — NVIDIA legal disclaimers and trademark notices.
