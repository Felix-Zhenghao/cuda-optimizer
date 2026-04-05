# Bounding-Box

Merged from: 5-5-3-1-Bounding-Box to 5-5-3-3-Out-of-Boundary-Access

Defines the tiled-mode Bounding Box (same dimensionality as tensor, 16-byte aligned and sized) with three access properties: dimension sizes, out-of-boundary mode, and traversal strides. Traversal stride controls element skip count per dimension (default 1; dimension-0 stride used for interleaved layouts). Out-of-bounds elements are filled with zero or a special OOB-NaN value.
