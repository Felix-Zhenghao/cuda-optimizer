# 9. Instruction Set

- **Format-and-Semantics-of-Instruction-Descriptions** — General PTX instruction format: operand notation, guard predicates, destination/source conventions, divergence semantics.
- **Predicated-Execution** — Integer/float/bit-size comparison operators and predicate manipulation via logic, conversion, and selection instructions.
- **Type-Information-for-Instructions-and-Operands** — Relaxed type-checking rules: truncation, zero/sign-extension, and valid type combination tables for ld/st/cvt.
- **Semantics** — Machine-specific 16-bit register promotion on 32-bit hardware; portable semantics require explicit masking.
- **Instructions** — Complete PTX instruction reference: arithmetic, FP, data movement, control flow, synchronization, matrix, tensor, and more.
