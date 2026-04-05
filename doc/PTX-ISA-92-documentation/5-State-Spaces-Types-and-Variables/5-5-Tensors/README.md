# Tensors

- **5-5-1-to-5-5-2-Tensor-Dimension-size-and-format** — Sub-byte element types with padding/packing rules, tiled vs. im2col access modes, and interleaved (NC/8, NC/16) layouts.
- **5-5-3-Tiled-Mode** — Bounding Box, traversal stride, out-of-boundary fill, and scatter4/gather4 variants for tiled tensor access.
- **5-5-4-im2colmode** — Im2col mode Bounding Box in DHW space, Pixels-per-Column, traversal stride, and out-of-boundary handling.
- **5-5-5-im2colwandim2colw128modes** — W-dimension-only `im2col::w` and `im2col::w::128` modes with wHalo and wOffset for convolution halo loading.
- **5-5-7-Swizzling-Modes** — Shared memory swizzling patterns (32B/64B/96B/128B with sub-modes) to optimize tensor data layout for access performance.
- **5-5-8-Tensor-map** — 128-byte opaque tensor descriptor in const/param/global space created by CUDA APIs and used by tensor copy instructions.
