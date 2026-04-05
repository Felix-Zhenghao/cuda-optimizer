# Grouped GEMM + Quant -- Unified (SM100)

Unified grouped GEMM with output quantization and per-row gating on Blackwell GPUs for MoE FC2/dFC1 workloads. Supports both dense and discrete (per-expert pointer) weight modes, block-scaled FP4/FP8 inputs, and row/column quantized outputs with optional scale factors.
