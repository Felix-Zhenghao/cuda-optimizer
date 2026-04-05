# NVIDIA cuDNN Frontend

Documentation for the cuDNN Frontend library -- a C++ and Python API for building and executing deep learning computation graphs on NVIDIA GPUs using cuDNN.

- **[Overview](Overview/)** — Frontend API layers, workflow (graph creation, validation, building, autotuning, execution), and full operation reference.
- **[Installation Guide](Installation-Guide/)** — Build from source with CMake or install via pip, with dependency and environment configuration.
- **[Samples](Samples/)** — C++ and Python sample code covering convolution, matmul, normalization, attention, and other operations.
- **[Release Notes](Releases-Notes/)** — Version history, new features, API changes, and known issues for cuDNN Frontend releases.
- **[Operations](Operations/)** — Attention, convolution, matmul, normalization, pointwise, reduction, resampling, block scaling, and more.
- **[Utilities](Utilities/)** — Torch custom ops integration, CUDA graphs, custom execution plans, AOT compilation, dynamic shapes, kernel cache.
- **[Frontend OSS APIs](Frontend-OSS-APIs/)** — Experimental Python APIs for fused attention, GEMM, normalization kernels on Blackwell GPUs.
- **[Core Concepts](Core-Concepts/)** — Tensors, data types, virtual tensors, and operation graph fundamentals.
- **[Graphs](Graphs/)** — Graph building, operation modes, tensor attributes, validation, autotuning, execution, and serialization.
- **[Hardware Forward Compatibility](Hardware-Forward-Compatibility/)** — Run cuDNN on newer GPU architectures without library recompilation.
- **[Odds and Ends](Odds-and-Ends/)** — Miscellaneous topics: heuristics, behavioral notes, errata, and runtime fusion engines.
- **[Debugging](Debugging/)** — Logging, error codes, numeric debugging tools, and graph visualization.
- **[Supported Products](Supported-Products/)** — Supported GPU architectures and CUDA compute capabilities.
- **[FAQs](FAQs/)** — Common questions on mixed precision, convolution algorithms, latency, and library linking.
- **[Support](Support/)** — NVIDIA developer portal, forums, and bug reporting guidance.
- **[Software License Agreement](Software-License-Agreement/)** — NVIDIA SDK license terms, distribution rights, and restrictions.
- **[Acknowledgements](Acknowledgements/)** — MIT License for cuDNN Frontend (Copyright 2020 NVIDIA).
- **[Notices](Notices/)** — Standard NVIDIA legal disclaimers and liability limitations.
