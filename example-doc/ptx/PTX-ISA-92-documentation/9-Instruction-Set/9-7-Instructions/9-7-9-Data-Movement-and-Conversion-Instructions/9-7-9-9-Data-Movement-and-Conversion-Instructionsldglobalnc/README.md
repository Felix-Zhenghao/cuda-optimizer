# 9.7.9.9. Data Movement and Conversion Instructions: ld.global.nc

The `ld.global.nc` instruction loads a register variable from global memory via the non-coherent texture cache, which may offer higher bandwidth on some architectures. Supports cache operators (.ca, .cg, .cs), eviction priorities, prefetch size hints, cache policies, and vector loads (.v2, .v4, .v8) with types up to .b128. Requires sm_32+. Introduced in PTX ISA 3.1.
