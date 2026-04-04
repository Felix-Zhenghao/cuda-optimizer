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
