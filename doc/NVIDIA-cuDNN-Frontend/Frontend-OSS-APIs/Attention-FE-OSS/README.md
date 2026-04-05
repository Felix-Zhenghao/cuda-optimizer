# SDPA Backward (SM100, D=256)

SDPA backward pass for head dimension 256 on Blackwell GPUs, computing attention gradients dQ, dK, dV using a two-kernel CUTE DSL implementation. Supports causal masking, sliding windows, variable-length batched sequences, and FP16/BF16 inputs with FP32 accumulation.
