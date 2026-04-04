# 9.7.16.10.4. Packing Formats of Elements in Tensor and Shared Memory

Defines packing formats for matrix elements in Tensor Memory and shared memory. Matrix D in Tensor Memory uses unpacked sub-word elements (one element per 32-bit word). Matrix A in Tensor Memory and matrices A/B in shared memory use packed formats where multiple sub-word elements share a 32-bit word.
