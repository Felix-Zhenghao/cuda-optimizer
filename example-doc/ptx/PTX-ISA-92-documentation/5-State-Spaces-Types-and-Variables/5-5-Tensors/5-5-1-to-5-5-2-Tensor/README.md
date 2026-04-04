# Tensor Dimension, Format, and Access Modes (5.5.1 to 5.5.2)

Merged from: 5-5-1-Tensor-Dimension-size-and-format to 5-5-2-Tensor-Access-Modes

Describes sub-byte tensor types (.b4x16, .b4x16_p64, .b6x16_p32, .b6p2x16) with their padding/alignment behavior during tensor copy operations between shared and global memory. Defines the two tensor access modes: tiled mode (preserves layout) and im2col mode (rearranges elements into columns).
