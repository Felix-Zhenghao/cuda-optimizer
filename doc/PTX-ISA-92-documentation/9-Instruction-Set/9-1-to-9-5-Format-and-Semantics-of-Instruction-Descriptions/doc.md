## 9.1. [Format and Semantics of Instruction Descriptions](#format-and-semantics-of-instruction-descriptions)

This section describes each PTX instruction. In addition to the name and the format of the
instruction, the semantics are described, followed by some examples that attempt to show several
possible instantiations of the instruction.

---

## 9.2. [PTX Instructions](#ptx-instructions)

PTX instructions generally have from zero to four operands, plus an optional guard predicate
appearing after an `@` symbol to the left of the `opcode`:

* `@pÂ Â  opcode;`
* `@pÂ Â  opcode a;`
* `@pÂ Â  opcode d, a;`
* `@pÂ Â  opcode d, a, b;`
* `@pÂ Â  opcode d, a, b, c;`

For instructions that create a result value, the `d` operand is the destination operand, while
`a`, `b`, and `c` are source operands.

The `setp` instruction writes two destination registers. We use a `|` symbol to separate
multiple destination registers.

```
setp.lt.s32  p|q, a, b;  // p = (a < b); q = !(a < b);
```

For some instructions the destination operand is optional. A *bit bucket* operand denoted with an
underscore (`_`) may be used in place of a destination register.

---

## 9.5. [Divergence of Threads in Control Constructs](#divergence-of-threads-in-control-constructs)

Threads in a CTA execute together, at least in appearance, until they come to a conditional control
construct such as a conditional branch, conditional function call, or conditional return. If threads
execute down different control flow paths, the threads are called *divergent*. If all of the threads
act in unison and follow a single control flow path, the threads are called *uniform*. Both
situations occur often in programs.

A CTA with divergent threads may have lower performance than a CTA with uniformly executing threads,
so it is important to have divergent threads re-converge as soon as possible. All control constructs
are assumed to be divergent points unless the control-flow instruction is marked as uniform, using
the `.uni` suffix. For divergent control flow, the optimizing code generator automatically
determines points of re-convergence. Therefore, a compiler or code author targeting PTX can ignore
the issue of divergent threads, but has the opportunity to improve performance by marking branch
points as uniform when the compiler or author can guarantee that the branch point is non-divergent.
