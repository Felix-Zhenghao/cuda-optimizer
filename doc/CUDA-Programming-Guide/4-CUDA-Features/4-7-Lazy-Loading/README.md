# 4.7. Lazy Loading

Lazy loading defers CUDA module initialization until first use, reducing startup time for applications using only a subset of kernels. Explains how it works, how to control it via CUDA_MODULE_LOADING, potential hazards around non-standard programming patterns, and performance measurement considerations for warmup.
