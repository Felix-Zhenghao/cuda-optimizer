# 9.7.9.18. Data Movement and Conversion Instructions: createpolicy

The `createpolicy` instruction creates a 64-bit opaque cache eviction policy for use with .L2::cache_hint qualifiers. Supports range-based policies (primary/secondary address ranges with different priorities), fraction-based policies (probabilistic eviction priority), and conversion from CUDA API access properties. Requires sm_80+. Introduced in PTX ISA 7.4.
