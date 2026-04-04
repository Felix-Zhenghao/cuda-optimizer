# 9.7.14.5.4. Matrix Fragments for mma.m8n8k32

Describes fragment distribution for `mma.m8n8k32` MMA operation. Covers sub-byte integer types (u4, s4) with s32 accumulator. Elements of matrices A, B, C, D are distributed so each thread holds specific b32 register fragments containing packed sub-byte elements.
