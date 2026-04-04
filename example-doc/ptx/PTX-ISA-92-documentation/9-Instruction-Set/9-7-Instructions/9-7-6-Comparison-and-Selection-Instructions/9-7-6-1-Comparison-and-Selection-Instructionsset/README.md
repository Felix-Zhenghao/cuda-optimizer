# 9.7.6.1. Comparison and Selection Instructions: set

The `set` instruction compares two numeric values using a relational operator (eq, ne, lt, le, gt, ge, etc.) and optionally combines the result with a predicate via a Boolean operator (and, or, xor). Writes 0xFFFFFFFF or 1.0f for true, 0x00000000 for false, to integer or float destinations respectively. Supports signed, unsigned, bit-size, and floating-point types including ordered and unordered NaN-aware comparisons. Introduced in PTX ISA 1.0.
