# Discrete Grouped GEMM + dSwiGLU (SM100)

Block-scaled grouped GEMM fused with dSwiGLU/dGeGLU backward epilogue on Blackwell GPUs for MoE workloads with per-expert weight allocations. Uses device pointer arrays for B and SFB tensors, computing backward gradients with optional output quantization and probability gradient accumulation.
