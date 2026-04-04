# PTX ISA 9.2 Documentation

- **1-Introduction** — Overview of PTX as a stable GPU programming model, ISA versioning, and document structure.
- **2-Programming-Model** — Thread hierarchy (CTAs, clusters, grids), memory hierarchy, and the data-parallel execution model.
- **3-PTX-Machine-Model** — SIMT multiprocessor architecture, independent thread scheduling, and on-chip shared memory.
- **4-Syntax** — PTX source format, comments, directive/instruction statements, identifiers, and constant expressions.
- **5-State-Spaces-Types-and-Variables** — Register/global/shared/local state spaces, fundamental and packed types, variables, tensors, and tensor maps.
- **6-Instruction-Operands** — Operand types, addresses, arrays, vectors, type conversions, rounding modifiers, and operand costs.
- **7-Abstracting-the-ABI** — Function declaration, parameter passing, call/return conventions, variadic functions, and alloca.
- **8-Memory-Consistency-Model** — Formal memory model: scopes, operation types, acquire/release, ordering axioms, and synchronization.
- **9-Instruction-Set** — Complete PTX instruction reference: arithmetic, logic, data movement, control flow, synchronization, matrix ops, and more.
- **10-Special-Registers** — Read-only registers: thread/block/grid IDs, warp info, clock, performance monitors, and hardware IDs.
- **11-Directives** — Module directives (.version, .target), kernel/function entry points, control flow, tuning, debugging, and linking.
- **12-Descriptions-ofpragmaStrings** — Compiler pragma strings for nounroll, unroll, and predicate control.
- **13-Release-Notes** — Per-version changelog for PTX ISA versions 2.0 through 9.2.
- **14-Notices** — Legal notices, NVIDIA trademarks, and third-party trademark acknowledgments.
