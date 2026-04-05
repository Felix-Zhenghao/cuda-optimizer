# Grouped GEMM + SwiGLU (SM100)

Legacy contiguous-only grouped GEMM fused with SwiGLU forward epilogue on Blackwell GPUs for MoE workloads. Uses block-scaled FP4/FP8 inputs with per-block scale factors and optional per-row gating. Superseded by the unified Grouped GEMM + GLU API.
