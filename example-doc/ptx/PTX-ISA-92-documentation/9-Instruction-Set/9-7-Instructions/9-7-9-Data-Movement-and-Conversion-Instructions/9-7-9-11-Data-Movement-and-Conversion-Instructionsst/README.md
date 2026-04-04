# 9.7.9.11. Data Movement and Conversion Instructions: st

The `st` instruction stores a register value to an addressable state space (.global, .local, .param, .shared). Supports memory ordering qualifiers (.weak, .volatile, .relaxed, .release), scope qualifiers, cache operations, eviction priorities, cache policies, and vector stores (.v2, .v4, .v8) with types up to .b128. The sink symbol '_' can be used in vector expressions. Introduced in PTX ISA 1.0.
