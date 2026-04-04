# 9.7.16.10.8.4. Sparsity Selector

Defines the sparsity metadata layout in Tensor Memory for tcgen05 sparse MMA. Metadata is a 2-bit selector per sub-chunk that identifies which columns contain non-zero elements. Includes detailed diagrams of metadata placement in Tensor Memory lanes for M=64 and M=256 across different kinds.
