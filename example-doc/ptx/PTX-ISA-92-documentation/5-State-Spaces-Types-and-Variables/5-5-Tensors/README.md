# 5.5. Tensors

- **5-5-1-to-5-5-2-Tensor** -- Sub-byte tensor types, padding/alignment, and tiled vs. im2col access modes.
- **5-5-3-Tiled-Mode** -- Tiled access with bounding box, traversal stride, OOB handling, and scatter4/gather4 modes.
- **5-5-4-im2colmode** -- Im2col mode: bounding box in DHW space, traversal stride, and out-of-boundary access.
- **5-5-5-im2colwandim2colw128modes** -- Im2col::w and im2col::w::128 modes with wHalo and wOffset for convolution.
- **5-5-6-Interleave-layout** -- Interleave layouts: none (NDHWC), 8-byte, and 16-byte channel interleaving.
- **5-5-7-Swizzling-Modes** -- Shared memory swizzling modes (32B/64B/96B/128B) with atomicity sub-modes.
- **5-5-8-Tensor-map** -- 128-byte opaque tensor-map object encoding tensor and access properties.
