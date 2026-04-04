# 9.7.14.5.2. Matrix Fragments for mma.m8n8k4 with .f64 Floating Point Type

Describes fragment distribution for `mma.m8n8k4` with f64 type. Each thread holds one f64 element of A, one of B, and two of C/D. Elements are distributed across the 32 threads based on thread grouping. Requires sm_80 or higher.
