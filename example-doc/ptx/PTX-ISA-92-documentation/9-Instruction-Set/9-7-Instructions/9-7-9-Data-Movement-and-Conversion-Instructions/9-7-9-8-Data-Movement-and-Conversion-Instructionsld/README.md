# 9.7.9.8. Data Movement and Conversion Instructions: ld

The `ld` instruction loads a register variable from an addressable state space (.const, .global, .local, .param, .shared). Supports memory ordering qualifiers (.weak, .volatile, .relaxed, .acquire), scope qualifiers (.cta, .cluster, .gpu, .sys), cache operations, eviction priorities, prefetch hints, and cache policies. Supports vector loads (.v2, .v4, .v8) and types up to .b128. Introduced in PTX ISA 1.0.
