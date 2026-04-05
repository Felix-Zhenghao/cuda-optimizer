# Grouped GEMM + dGLU (SM100)

Unified grouped GEMM fused with dSwiGLU or dGeGLU backward epilogue on Blackwell GPUs for MoE workloads. Supports both dense and discrete (per-expert pointer) weight modes, block-scaled FP4/FP8 inputs, optional output quantization with row/column scale factors.
