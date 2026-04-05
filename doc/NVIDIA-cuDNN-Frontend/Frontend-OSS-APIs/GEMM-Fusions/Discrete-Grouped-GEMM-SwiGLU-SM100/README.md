# Discrete Grouped GEMM + SwiGLU (SM100)

Block-scaled grouped GEMM fused with SwiGLU/GeGLU epilogue on Blackwell GPUs for MoE workloads with per-expert weight allocations. Uses device pointer arrays for B and SFB tensors, supporting FP4/FP8 inputs with optional output quantization and per-row gating.
