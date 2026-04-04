# 9.7.8.6. Logic and Shift Instructions: lop3

The `lop3` instruction performs an arbitrary logical operation on three 32-bit inputs using an 8-bit lookup table (immLut) encoding up to 256 distinct Boolean functions. Optionally computes a predicate result by applying a Boolean operation (.or/.and) between the destination and a predicate operand. Requires sm_50+; BoolOp qualifier requires sm_70+. Introduced in PTX ISA 4.3.
