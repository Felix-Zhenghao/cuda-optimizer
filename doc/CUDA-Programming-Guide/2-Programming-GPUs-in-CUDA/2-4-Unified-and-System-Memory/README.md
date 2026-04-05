# 2.4. Unified and System Memory

Covers CUDA memory abstractions for heterogeneous systems: unified virtual address space (single pointer space across CPU and all GPUs), unified memory paradigms (full with hardware ATS, full with software HMM, limited on Windows/Tegra), `cudaMemAdvise`/`cudaMemPrefetchAsync` performance hints, and page-locked host memory including mapped (zero-copy) memory via `cudaMallocHost`, `cudaHostAlloc`, and `cudaHostRegister`.
