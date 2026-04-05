# Grouped GEMM + dSwiGLU (SM100)

Legacy contiguous-only grouped GEMM fused with dSwiGLU backward epilogue on Blackwell GPUs for MoE workloads. Uses block-scaled FP4/FP8 inputs with per-block scale factors, producing row/column quantized outputs. Superseded by the unified Grouped GEMM + dGLU API.
