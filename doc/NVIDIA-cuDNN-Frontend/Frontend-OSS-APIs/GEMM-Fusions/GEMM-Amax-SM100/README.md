# GEMM + Amax (SM100)

Persistent batched dense GEMM on Blackwell GPUs with block-scaled FP8/FP4 inputs and per-block scale factors, producing the full GEMM output plus a global amax reduction. Implemented with CUTLASS/CUTE, supports configurable tiling and cluster shapes.
