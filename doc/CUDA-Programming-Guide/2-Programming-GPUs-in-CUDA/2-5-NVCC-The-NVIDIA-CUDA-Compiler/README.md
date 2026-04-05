# 2.5. NVCC: The NVIDIA CUDA Compiler

Reference for the `nvcc` compiler toolchain: source file extensions (`.cu`, `.cuh`), the host/device code split compilation workflow (PTX → Cubin → Fatbin), GPU architecture targeting (`-arch`, `-gencode`), separate compilation with `-rdc=true` and Link-Time Optimization, and key compiler flags for language standard, debugging (`-G`, `-lineinfo`), optimization, profiling, and fatbin compression.
