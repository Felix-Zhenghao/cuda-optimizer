# Grouped GEMM + GLU (SM100)

Unified grouped GEMM fused with SwiGLU or GeGLU forward epilogue on Blackwell GPUs for MoE workloads. Supports both dense and discrete weight modes, block-scaled FP4/FP8 inputs, optional bias, per-row gating, and output quantization with row/column scale factors.
