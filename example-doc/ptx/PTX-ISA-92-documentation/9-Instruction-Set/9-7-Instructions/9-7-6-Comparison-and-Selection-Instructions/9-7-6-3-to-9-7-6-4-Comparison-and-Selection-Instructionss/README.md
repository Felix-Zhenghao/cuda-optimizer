# 9.7.6.3-4. Comparison and Selection Instructions: selp, slct

Covers two conditional selection instructions. `selp` selects between two source operands based on a predicate value (true selects `a`, false selects `b`). `slct` selects between two operands based on the sign of a third operand (non-negative selects `a`, negative selects `b`). Both support integer and floating-point destination types. Introduced in PTX ISA 1.0.
