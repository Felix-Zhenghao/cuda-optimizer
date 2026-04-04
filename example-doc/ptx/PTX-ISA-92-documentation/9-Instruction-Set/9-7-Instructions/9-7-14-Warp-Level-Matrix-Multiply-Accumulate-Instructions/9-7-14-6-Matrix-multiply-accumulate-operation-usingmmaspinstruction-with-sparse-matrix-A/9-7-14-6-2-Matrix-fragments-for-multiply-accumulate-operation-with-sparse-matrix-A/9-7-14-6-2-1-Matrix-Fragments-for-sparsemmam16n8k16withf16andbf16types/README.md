# 9.7.14.6.2.1. Matrix Fragments for sparse mma.m16n8k16 with .f16 and .bf16 Types

Describes per-thread fragment distribution for sparse `mma.m16n8k16` with f16/bf16 types. Matrix A stores only non-zero elements (2:4 sparsity). Includes fragment layouts for compressed A, full B, accumulator C/D, and sparsity metadata across warp threads.
