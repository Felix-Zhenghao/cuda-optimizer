# 4.20. Driver Entry Point Access

Driver Entry Point Access APIs allow retrieval of CUDA driver function pointers for version-aware dynamic dispatch. Covers typedef declarations, cuGetProcAddress usage, version targeting, flags for default vs per-thread stream variants, and diagnosing failures (version insufficient vs symbol not found return codes).
