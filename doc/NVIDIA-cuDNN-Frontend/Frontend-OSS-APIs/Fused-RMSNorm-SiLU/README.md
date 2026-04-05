# Fused RMSNorm + SiLU

Single-kernel fusion of RMS normalization followed by SiLU activation, optimized for the WAN VAE decoder pattern on Blackwell GPUs. Supports bf16, FP8, and NVFP4 outputs across SM80 to SM103 architectures, compiled at runtime via NVRTC with sweep-tuned configurations for SM100.
