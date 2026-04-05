# 2.1 Intro to CUDA C++ — Kernels and Memory

Introduces CUDA C++ fundamentals: the `nvcc` compiler, kernel declaration with `__global__`, triple-chevron launch syntax, thread/grid index intrinsics, bounds checking, and two memory management approaches (unified memory via `cudaMallocManaged` and explicit management via `cudaMalloc`/`cudaMemcpy`), illustrated with a complete vector-addition example.
