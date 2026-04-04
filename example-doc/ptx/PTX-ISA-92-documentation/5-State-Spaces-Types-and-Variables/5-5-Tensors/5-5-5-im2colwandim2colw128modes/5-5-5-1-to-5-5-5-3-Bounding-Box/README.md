# Im2col::w/w::128 Bounding Box, Traversal Stride, and wHalo (5.5.5.1 to 5.5.5.3)

Merged from: 5-5-5-1-Bounding-Box to 5-5-5-3-wHalo

Describes im2col::w and im2col::w::128 modes where D/H bounding box dimensions are 1 and access occurs along W dimension. Covers W-dimension bounding box corners, traversal stride behavior, and the wHalo argument for loading convolution filter halo elements at row boundaries (per-row in w mode, per-32-elements in w::128 mode).
