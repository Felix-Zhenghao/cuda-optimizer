## 6.1. [Operand Type Information](#operand-type-information)

All operands in instructions have a known type from their declarations. Each operand type must be
compatible with the type determined by the instruction template and instruction type. There is no
automatic conversion between types.

The bit-size type is compatible with every type having the same size. Integer types of a common size
are compatible with each other. Operands having type different from but compatible with the
instruction type are silently cast to the instruction type.

---

## 6.2. [Source Operands](#source-operands)

The source operands are denoted in the instruction descriptions by the names `a`, `b`, and
`c`. PTX describes a load-store machine, so operands for ALU instructions must all be in variables
declared in the `.reg` register state space. For most operations, the sizes of the operands must
be consistent.

The `cvt` (convert) instruction takes a variety of operand types and sizes, as its job is to
convert from nearly any data type to any other data type (and size).

The `ld`, `st`, `mov`, and `cvt` instructions copy data from one location to
another. Instructions `ld` and `st` move data from/to addressable state spaces to/from
registers. The `mov` instruction copies data between registers.

Most instructions have an optional predicate guard that controls conditional execution, and a few
instructions have additional predicate source operands. Predicate operands are denoted by the names
`p`, `q`, `r`, `s`.

---

## 6.3. [Destination Operands](#destination-operands)

PTX instructions that produce a single result store the result in the field denoted by `d` (for
destination) in the instruction descriptions. The result operand is a scalar or vector variable in
the register state space.

---

## 6.6. [Operand Costs](#operand-costs)

Operands from different state spaces affect the speed of an operation. Registers are fastest, while
global memory is slowest. Much of the delay to memory can be hidden in a number of ways. The first
is to have multiple threads of execution so that the hardware can issue a memory operation and then
switch to other execution. Another way to hide latency is to issue the load instructions as early as
possible, as execution is not blocked until the desired result is used in a subsequent (in time)
instruction. The register in a store operation is available much more
quickly. [Table 19](#operand-costs-cost-estimates-for-sccessing-state-spaces) gives estimates of the
costs of using different kinds of memory.

Table 19 Cost Estimates for Accessing State-Spaces

| Space | Time | Notes |
| --- | --- | --- |
| Register | 0 |  |
| Shared | 0 |  |
| Constant | 0 | Amortized cost is low, first access is high |
| Local | > 100 clocks |  |
| Parameter | 0 |  |
| Immediate | 0 |  |
| Global | > 100 clocks |  |
| Texture | > 100 clocks |  |
| Surface | > 100 clocks |  |
