# 9.4. Type Information for Instructions and Operands

Covers relaxed type-checking rules when operand sizes exceed instruction-type sizes. For `ld`, `st`, and `cvt`, source operands wider than the instruction type are truncated (chopped), while destination operands are zero-extended or sign-extended. Tables 27 and 28 detail the full source/destination type compatibility matrix for bit-size, integer, and floating-point types.
