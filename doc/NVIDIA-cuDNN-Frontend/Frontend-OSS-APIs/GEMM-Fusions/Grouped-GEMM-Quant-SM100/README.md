# Grouped GEMM + Quant (SM100)

Legacy dense-only grouped GEMM with output quantization and per-row gating on Blackwell GPUs for MoE FC2/dFC1 workloads. Uses block-scaled FP4/FP8 inputs, producing row/column quantized outputs with optional scale factors. Superseded by the unified Grouped GEMM + Quant API.
