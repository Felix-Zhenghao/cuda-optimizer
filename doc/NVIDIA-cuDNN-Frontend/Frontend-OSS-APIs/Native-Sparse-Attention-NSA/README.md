# Native Sparse Attention (NSA)

Implements the NSA sparse attention mechanism for Blackwell GPUs, combining Selection Attention, Compression Attention, Sliding Window Attention, and Top-K Reduction components. Supports GQA/MQA, variable-length batched sequences, and FP8/FP16/BF16 inputs via CUTLASS/CUTE kernels.
