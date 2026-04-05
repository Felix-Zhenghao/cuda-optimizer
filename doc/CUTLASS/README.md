# Welcome to CUTLASS

NVIDIA CUTLASS documentation covering the high-performance GEMM and linear algebra template library for NVIDIA GPUs, including CuTe DSL, architecture support from Volta to Blackwell, and comprehensive API references.

- **Overview** — Project overview: CUTLASS 4.4.2 purpose, supported data types, architecture support, performance benchmarks.
- **Quick-Start-Guide** — Pip-based installation for CUTLASS DSL 4.4 with CUDA 12.9/13.1 and Python dependencies.
- **Getting-Started** — Build instructions, fundamental types, GEMM heuristics, IDE setup, and functionality tables.
- **Code-Organization** — Repository layout: template library, CuTe core, instance library, profiler, utilities, examples, tests.
- **Efficient-GEMM-in-CUDA** — Hierarchical GEMM implementation: threadblock tiling, warp-level GEMM, pipelining, split-K, warp specialization.
- **CuTe** — Core Layout and Tensor abstractions: quickstart, layout algebra, algorithms, MMA atoms, GEMM tutorial, TMA.
- **CuTe-DSL** — Python-embedded DSL: JIT compilation, control flow, code generation, AOT compilation, debugging, autotuning.
- **CuTe-DSL-API** — API reference for CuTe DSL modules: arch built-ins, GPU-specific MMA/copy ops, pipelines, utilities.
- **CUTLASS-2x** — CUTLASS 2.x API: GEMM API, layouts/tensors, tile iterators, utilities.
- **CUTLASS-3x** — CUTLASS 3.x API: design philosophy, GEMM API, backwards compatibility from 2.x.
- **CUTLASS-Convolution** — Implicit GEMM convolution: activation/filter iterators, Tensor Cores, permuted shared memory.
- **CUTLASS-Profiler** — Command-line profiling: instantiation levels, mixed dtype kernels, CLI usage.
- **Blackwell-Specific** — Blackwell cluster launch control and SM100/SM120 GEMM instructions with block scaling.
- **Dependent-Kernel-Launch** — Programmatic Dependent Launch (PDL) for overlapping kernel execution on Hopper/Blackwell.
- **Grouped-Kernel-Schedulers** — Persistent threadblock schedulers for batched GEMM and Rank2K with load balancing.
- **Synchronization-primitives** — Hopper synchronization: cluster sync, barriers, async pipeline producer-consumer patterns.
- **Functionality** — Supported MMA operations across Ampere, Hopper, Blackwell architectures and data types.
- **Limitations** — CuTe DSL constraints: static/dynamic values, control flow, OOP, debugging limitations.
- **Changelog** — Release history across CUTLASS 1.x, 2.x, 3.x, and 4.x version lines.
- **FAQs** — Common questions about DSL vs C++ templates, migration, architecture support, debugging.
- **Software-License-Agreement** — NVIDIA CUTLASS DSLs license terms, restrictions, and legal provisions.
