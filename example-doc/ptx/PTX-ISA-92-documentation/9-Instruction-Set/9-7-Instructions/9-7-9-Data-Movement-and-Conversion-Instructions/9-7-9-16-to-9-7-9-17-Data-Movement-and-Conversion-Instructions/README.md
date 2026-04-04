# 9.7.9.16-17. Data Movement and Conversion Instructions: applypriority, discard

Covers two cache management instructions. `applypriority` applies a cache eviction priority to a 128-byte aligned address range in a specified cache level. `discard` writes unstable indeterminate values to a 128-byte address range, hinting that cached data can be destructively discarded without write-back. Both require sm_80+. Introduced in PTX ISA 7.4.
