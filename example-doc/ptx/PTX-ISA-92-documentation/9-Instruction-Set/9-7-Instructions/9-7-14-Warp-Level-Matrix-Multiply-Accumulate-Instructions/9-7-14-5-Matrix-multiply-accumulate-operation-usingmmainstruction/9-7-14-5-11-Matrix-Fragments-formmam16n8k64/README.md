# 9.7.14.5.11. Matrix Fragments for mma.m16n8k64

Describes fragment distribution for `mma.m16n8k64` MMA operation. Covers sub-byte integer types (u4, s4) and sub-byte floating-point type (e2m1) with block scaling. Each thread holds multiple b32 register fragments containing packed elements for the 16x8x64 operation shape.
