# Deprecation Policy

## Purpose

The goal of this policy is to evolve the DSL and its APIs while keeping user
programs stable. Features or APIs are deprecated only when they are redundant,
unsafe, or block better designs.

## Deprecation Process

**Step 1 â Soft Deprecation**

When a feature is considered for removal, it is first annotated with the
`@deprecated` decorator or `DeprecationWarning` and documented with a
suggested alternative. At this stage, the feature continues to work normally.

Users are encouraged to provide feedback and describe their use cases.
If there is strong justification, we may keep or redesign the feature.

**Step 2 â Removal (the subsequent release)**

If no valid use cases remain, the deprecated feature will be removed in the
following **minor** release.

Note

The release version follows the format `<major>.<minor>.<patch>`.

## Communication

All deprecations are announced through:

* This page
* In-code warning messages

## Soft Deprecations

**Version 4.2.1**

* `cute.arch.warpgroup_reg_alloc` and `cute.arch.warpgroup_reg_dealloc`
  â Scheduled for deprecation. Use `cute.arch.setmaxregister_increase` and `cute.arch.setmaxregister_decrease` instead.
* `alignment` argument in `CooperativeGroup` constructor
  â Scheduled for deprecation. It was unused; no replacement is suggested.

## Deprecated Features

*(None currently.)*

---

# Educational Notebooks

A number of notebooks for educational purposes are provided in the [CUTLASS GitHub repository](https://github.com/NVIDIA/cutlass).
A list with handful links is given below:

* ["Hello world"](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/hello_world.ipynb)
* [Printing](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/print.ipynb)
* [Data Types Basics](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/data_types.ipynb)
* [Tensors](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensor.ipynb)
* [The TensorSSA Abstraction](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensorssa.ipynb)
* [Layout Algebra](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/cute_layout_algebra.ipynb)
* [Element-wise Add Tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb)
* [Using CUDA Graphs](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/cuda_graphs.ipynb)

---

# Talks and Presentations

This page collects talks, presentations, and other resources related to CuTe DSL
and CUTLASS Python infrastructure.

## Conference Talks

**CuTeDSL: CUTLASS Python DSL Infrastructure** â *LLVM 2025*

An introduction to the CuTe DSL architecture, covering the hybrid AST-rewrite and
tracing approach, MLIR code generation, and integration with CUTLASS.

* [LLVM Video](https://www.youtube.com/watch?v=5NXd6MbKYNQ)
* [Slides (PDF)](https://llvm.org/devmtg/2025-10/slides/technical_talks/ozen.pdf)

---

**Enable Tensor Core Programming in Python with CUTLASS 4.0** â *GTC 2025*

Learn how to leverage Tensor Cores directly from Python using CUTLASS 4.0's
new DSL front-end, enabling rapid kernel development without writing CUDA C++.

* [GTC Video](https://www.nvidia.com/en-us/on-demand/session/gtc25-s74639/)
