# CuTe

- **00_quickstart**: Getting started guide with system requirements, build instructions, and library overview.
- **01_layout**: Core Layout abstraction mapping coordinates to indices with hierarchical shapes and strides.
- **02_layout_algebra**: Advanced Layout operations including coalesce, composition, complement, division, and product.
- **03_tensor**: Tensor container composing iterators with Layouts, supporting tiling, slicing, and partitioning.
- **04_algorithms**: Generic tensor algorithms: copy, gemm, axpby, fill, and clear.
- **0t_mma_atom**: Hardware MMA instruction support with Operation, Traits, Atom, and TiledMMA abstractions.
- **0x_gemm_tutorial**: End-to-end GEMM tutorial building dense matrix multiply from CuTe primitives.
- **0y_predication**: Predication techniques for handling imperfect tiling with bounds-checked coordinate tensors.
- **0z_tma_tensors**: Advanced TMA Tensor types using ArithmeticTuple iterators for Hopper TMA instructions.
