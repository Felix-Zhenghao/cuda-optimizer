# 5. State Spaces, Types, and Variables

- **5.1. State Spaces** — PTX memory hierarchy: registers, special registers, global, local, shared, parameter, constant (deprecated banked), and texture (deprecated) state spaces.
- **5.2. Types** — Fundamental types, alternate floating-point formats (bf16, e4m3, tf32, etc.), fixed-point s2f6, and packed data types for SIMD-style instruction operands.
- **5.3. Texture Sampler and Surface Types** — Opaque `.texref`, `.samplerref`, `.surfref` type fields including dimensions, channel format, sampler properties, and OpenCL channel enumerations.
- **5.4. Variables** — Variable declaration syntax, vector and array types, initializers with generic/mask operators, alignment, parameterized names, and `.managed`/`.unified` attributes.
- **5.5. Tensors** — Tensor map descriptor, access modes (tiled, im2col, im2col::w), bounding box parameters, traversal strides, interleaved layouts, and shared memory swizzling modes.
