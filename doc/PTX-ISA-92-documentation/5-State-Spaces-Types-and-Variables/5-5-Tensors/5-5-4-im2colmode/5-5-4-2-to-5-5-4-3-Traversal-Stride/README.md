# Traversal-Stride

Merged from: 5-5-4-2-Traversal-Stride to 5-5-4-3-Out-of-Boundary-Access

In im2col mode, traversal stride strides the D, H, and W dimensions but does not change total element count (set by Pixels-per-Column). Out-of-boundary access occurs when Pixels-per-Column exceeds available image-batch pixels; handled by zero fill or OOB-NaN fill depending on the Fill-Mode specified in the tensor descriptor.
