# 5.5.3. Tiled Mode

Covers tiled tensor access mode with bounding box concepts: dimension sizes (must be 16-byte multiples), traversal strides for element skipping, and out-of-boundary handling (zero fill or OOB-NaN fill). Also describes tiled::scatter4 and tiled::gather4 modes where four bounding boxes are formed from four request coordinates.
