# 5.5.8. Tensor-map

A tensor-map is a 128-byte opaque descriptor in `.const`, `.param`, or `.global` space that encodes tensor properties (dimensions, format, access mode) and associated access parameters. Tensor-maps are created using CUDA runtime APIs and passed to PTX tensor copy instructions to specify the source or destination tensor layout.
