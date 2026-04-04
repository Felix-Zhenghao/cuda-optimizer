# Im2col Traversal Stride and OOB Access (5.5.4.2 to 5.5.4.3)

Merged from: 5-5-4-2-Traversal-Stride to 5-5-4-3-Out-of-Boundary-Access

Explains that im2col traversal stride affects element spacing along D/H/W dimensions but not total pixel count (controlled by Pixels-per-Column). Covers out-of-boundary handling when requested pixels exceed available image batch pixels, with zero fill or OOB-NaN fill modes.
