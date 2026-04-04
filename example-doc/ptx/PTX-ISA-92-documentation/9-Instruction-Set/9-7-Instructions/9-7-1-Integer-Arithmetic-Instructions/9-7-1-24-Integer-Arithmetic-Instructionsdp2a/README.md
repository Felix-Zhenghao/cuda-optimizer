# 9.7.1.24 Integer Arithmetic Instructions: dp2a

Documents the PTX `dp2a` instruction for two-way 16-bit to 8-bit dot product-accumulate. Takes packed 16-bit values from operand `a` and packed 8-bit values from operand `b`, with `.lo`/`.hi` mode selecting which half of `b` to use. Requires sm_61 or higher.
