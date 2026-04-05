# Tensor-Dimension-size-and-format

Merged from: 5-5-1-Tensor-Dimension-size-and-format to 5-5-2-to-5-5-6-Tensor-Access-Modes

Covers sub-byte tensor element types (`.b4x16`, `.b4x16_p64`, `.b6x16_p32`, `.b6p2x16`) with their global-to-shared memory padding and packing rules, and the two tensor access modes: tiled (preserves source multi-dimensional layout at destination) and im2col (rearranges bounding-box elements into columns). Also covers interleaved layout (NDHWC, NC/8, NC/16) for 3D–5D tensors.
