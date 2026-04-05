# GEMM + SwiGLU (SM100)

Persistent batched dense GEMM fused with a SwiGLU epilogue on Blackwell GPUs, producing both the full GEMM output and a SwiGLU-projected tensor in a single pass. Supports standard high-precision and block-scaled quantized (FP4/FP8) modes via CUTLASS/CUTE.
