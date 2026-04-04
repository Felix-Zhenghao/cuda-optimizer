# 5.5.8. Tensor Map

Describes the tensor-map: a 128-byte opaque object in .const, .param (kernel parameter), or .global space that encodes tensor properties and access configuration. Tensor maps are created using CUDA APIs and passed to PTX tensor instructions to describe the data layout and access patterns.
